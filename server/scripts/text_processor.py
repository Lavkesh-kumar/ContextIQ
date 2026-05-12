from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextProcessor:

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # change to a another model if needed

    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.astype(np.float32)

    def process_text(self, text: str) -> Tuple[List[str], np.ndarray]:
        chunks = self.chunk_text(text)
        embeddings = self.get_embeddings(chunks)
        return chunks, embeddings