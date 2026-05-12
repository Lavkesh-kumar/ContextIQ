import faiss
import numpy as np
from typing import Tuple
import os


class VectorStore:
    
    def __init__(self):
        self.index = None
        self.dimension = None
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        
        self.dimension = embeddings.shape[1]
        n_docs = embeddings.shape[0]
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        return self.index
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        k = min(k, self.index.ntotal)
        faiss.normalize_L2(query_embedding)
        distances,indices = self.index.search(query_embedding, k) 
        print(f"Distances: {distances}, Indices: {indices}")
        return distances, indices
    
    # def save_index(self, filepath: str):
    #     os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #     faiss.write_index(self.index, filepath)
    
    # def load_index(self, filepath: str):
    #     self.index = faiss.read_index(filepath)
    #     self.dimension = self.index.d

    def reset(self):
        self.index = None
        self.dimension = None