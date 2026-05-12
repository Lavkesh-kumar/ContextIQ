import threading
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdflib import Graph
import os

from scripts.document_parser import DocumentParser
from scripts.rag_engine import RAGEngine
from scripts.text_processor import TextProcessor
from scripts.vector_store import VectorStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


ALLOWED_FILE_TYPES = {"pdf", "docx", "pptx", "xlsx", "txt"}


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    num_chunks: int
    created_at: str


class QueryRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    k: int = Field(3, ge=1, le=20)


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    sources: List[str]
    num_sources: int


class _DocCacheEntry:
    def __init__(
        self,
        *,
        filename: str,
        file_type: str,
        chunks: List[str],
        vector_store: VectorStore,
        graph: Graph,
        created_at: str,
    ):
        self.filename = filename
        self.file_type = file_type
        self.chunks = chunks
        self.vector_store = vector_store
        self.graph = graph
        self.created_at = created_at


class _AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.rag_engine: Optional[RAGEngine] = None
        self.doc_cache: Dict[str, _DocCacheEntry] = {}

    def get_rag_engine(self) -> RAGEngine:
        with self.lock:
            if self.rag_engine is None:
                # Default path updated for Windows local run
                model_path = os.getenv("CONTEXTIQ_MODEL_PATH", "./model/phi-3-openvino")
                device = os.getenv("CONTEXTIQ_DEVICE", "CPU")
                max_tokens = int(os.getenv("CONTEXTIQ_MAX_NEW_TOKENS", "512"))
                self.rag_engine = RAGEngine(model_path=model_path, device=device, max_tokens=max_tokens)
            return self.rag_engine

    def get_doc(self, doc_id: str) -> _DocCacheEntry:
        with self.lock:
            entry = self.doc_cache.get(doc_id)
            if entry is None:
                raise KeyError(doc_id)
            return entry


state = _AppState()

app = FastAPI(
    title="ContextIQ API",
    version="1.0.0",
    description="Upload documents and query them using the ContextIQ RAG engine.",
)

cors_origins_raw = os.getenv("CONTEXTIQ_CORS_ALLOW_ORIGINS", "*").strip()
allow_origins = ["*"] if cors_origins_raw == "*" else [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/documents", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> IngestResponse:
    filename = file.filename or "uploaded"
    file_type = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "").strip()

    if file_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file_type}'. Allowed: {sorted(ALLOWED_FILE_TYPES)}")

    if chunk_size < 100 or chunk_size > 2000:
        raise HTTPException(status_code=400, detail="chunk_size must be between 100 and 2000")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be >= 0 and < chunk_size")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        text = DocumentParser.parse_document(BytesIO(payload), file_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, embeddings = processor.process_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks produced from document")

    vector_store = VectorStore()
    vector_store.create_index(embeddings)

    rag = state.get_rag_engine()
    graph = rag.build_knowledge_graph(chunks)

    doc_id = uuid.uuid4().hex
    created_at = _utc_now_iso()

    with state.lock:
        if len(state.doc_cache) >= 10:
            state.doc_cache.pop(next(iter(state.doc_cache)))
        state.doc_cache[doc_id] = _DocCacheEntry(
            filename=filename,
            file_type=file_type,
            chunks=chunks,
            vector_store=vector_store,
            graph=graph,
            created_at=created_at,
        )

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        num_chunks=len(chunks),
        created_at=created_at,
    )


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        entry = state.get_doc(req.doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="doc_id not found. Please re-upload your document.")

    try:
        rag = state.get_rag_engine()
        result: Dict[str, Any] = rag.query(
            question=req.question,
            chunks=entry.chunks,
            vector_store=entry.vector_store,
            graph=entry.graph,
            k=req.k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        doc_id=req.doc_id,
        question=req.question,
        answer=str(result.get("answer", "")),
        sources=list(result.get("sources", [])),
        num_sources=int(result.get("num_sources", 0)),
    )
