---
title: ContextIQ
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ContextIQ - A Local RAG Based Secure Document Q&A System

ContextIQ is a fully local Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask AI-powered questions about their content — without using any external APIs. All processing happens on your machine.

## Features

- **Multi-format Document Support**: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), and Text (.txt)
- **Two-Stage Retrieval**: Vector similarity search (FAISS) + Knowledge Graph neighbor expansion for richer context
- **AI-Powered Answers**: Leverages OpenVINO Phi-3 mini (int4 quantized) for fast local inference
- **Client-Server Architecture**: FastAPI backend + Streamlit frontend
- **Source Attribution**: Shows which document chunks were used to generate each answer
- **Conversation History**: Tracks all Q&A pairs in the session

## Architecture

```
User Upload → Document Parser → Text Chunker → Embedding Generator → FAISS Index
                                                                            ↓
User Question → Query Embedder → FAISS Search → Graph Expansion → LLM (Phi-3) → Answer
```

### RAG Pipeline Detail

1. Document is parsed and split into chunks (default: 500 chars, 100 overlap)
2. Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`
3. Embeddings are stored in a FAISS index
4. A knowledge graph (RDFLib) links adjacent chunks as neighbors
5. At query time: top-k chunks are retrieved via vector search, then expanded with graph neighbors
6. The expanded context is passed to OpenVINO Phi-3 to generate the final answer

## Project Structure

```
ContextIQ/
│
├── client/
│   └── app.py                  # Streamlit frontend (port 8501)
│
├── server/
│   ├── main.py                 # FastAPI backend (port 7860)
│   ├── download_model.py       # Script to download Phi-3 OpenVINO model
│   └── scripts/
│       ├── document_parser.py  # PDF, DOCX, XLSX, PPTX, TXT parsers
│       ├── text_processor.py   # Chunking + embedding generation
│       ├── vector_store.py     # FAISS index wrapper
│       └── rag_engine.py       # RAG pipeline + knowledge graph + LLM
│
├── model/
│   └── phi-3-openvino/         # ❌ NOT included in GitHub — must be downloaded
│
├── requirements.txt
├── Dockerfile
├── start.sh
└── README.md
```

## Prerequisites

- Python 3.8+
- ~4 GB disk space for the Phi-3 OpenVINO model
- Internet connection for first-time model and embedding downloads

---

## Method 1: Local Setup (Recommended for Development)

### Step 1 — Clone the repository

```bash
git clone https://github.com/satheeshbhukya/ContextIQ.git
cd ContextIQ
```

### Step 2 — Create and activate a virtual environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Phi-3 OpenVINO model

Run this from the `server/` directory:

```bash
cd server
python download_model.py
cd ..
```

This downloads `OpenVINO/Phi-3-mini-4k-instruct-int4-ov` from Hugging Face into `model/phi-3-openvino/`. The model is ~2 GB. This step is only needed once.

### Step 5 — Start the FastAPI server

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
```

The server starts at `http://localhost:7860`. The model loads on the first document upload (not at startup).

### Step 6 — Start the Streamlit client (new terminal)

```bash
cd client
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}` when the server is running.

### `POST /v1/documents`
Upload and index a document.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Document file (pdf, docx, xlsx, pptx, txt) |
| `chunk_size` | int | 500 | Characters per chunk (100–2000) |
| `chunk_overlap` | int | 100 | Overlap between chunks (0 to chunk_size-1) |

Response:
```json
{
  "doc_id": "abc123",
  "filename": "report.pdf",
  "file_type": "pdf",
  "num_chunks": 42,
  "created_at": "2024-01-01T00:00:00+00:00"
}
```

### `POST /v1/query`
Query an indexed document.

```json
{
  "doc_id": "abc123",
  "question": "What are the key findings?",
  "k": 3
}
```

Response:
```json
{
  "doc_id": "abc123",
  "question": "What are the key findings?",
  "answer": "The key findings are...",
  "sources": ["chunk text 1", "chunk text 2"],
  "num_sources": 2
}
```

---

## Using the Streamlit UI

1. Upload a document from the sidebar (PDF, DOCX, PPTX, XLSX, TXT)
2. Wait for processing — a success message shows the chunk count
3. Type a question and click **Ask**
4. View the answer with expandable source chunks
5. Use **Clear Chat History** to reset Q&A history
6. Use **Reset Document** to upload a new document

The sidebar also shows live API status (Online/Unreachable).

---

## Configuration

In the Streamlit sidebar you can adjust:
- **Number of chunks to retrieve** (1–10): Higher = more context passed to LLM, but slower

---

## Example Use Cases

1. **Research Papers**: Ask about methodologies, findings, and conclusions from PDFs
2. **Legal Documents**: Query contracts and agreements for specific clauses
3. **Business Reports**: Extract insights from quarterly reports and presentations
4. **Technical Documentation**: Search for procedures and configurations
5. **Academic Notes**: Ask questions about lecture slides and study materials

---

## Technical Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Document Parsing | PyMuPDF, python-docx, python-pptx, pandas |
| Text Chunking | LangChain text splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (CPU) |
| Knowledge Graph | RDFLib |
| LLM | OpenVINO Phi-3 mini int4 (`openvino-genai`) |

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────┐
│  QUICK REFERENCE                                        │
├─────────────────────────────────────────────────────────┤
│  Start Server:   uvicorn main:app --port 7860           │
│  Start Client:   streamlit run app.py                   │
│  Upload Doc:     Sidebar → Browse files → Select file   │
│  Ask Question:   Type → Click Ask                       │
│  Adjust Chunks:  Drag slider (1-10)                     │
│  Clear History:  Click "Clear Chat History"             │
│  New Document:   Click "Reset Document"                 │
│  Stop:           Ctrl+C in terminal                     │
└─────────────────────────────────────────────────────────┘
```

---

## Demo

![Demo Image 1](demo/image1.png)

![Demo Image 2](demo/image2.png)

---

## License

MIT License - feel free to use this project for personal or commercial purposes.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [satheeshbhukyaa@gmail.com]

---

**⭐ If you find this project useful, please consider giving it a star!**
