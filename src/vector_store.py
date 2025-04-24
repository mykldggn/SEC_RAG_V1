"""
FAISS-based vector store adapter.
"""
import numpy as np
from typing import Any, List, Dict
from .config import VECTOR_DB

class VectorStore:
    def __init__(self, backend: str = None, **kwargs):
        # For now, only FAISS is supported
        import faiss
        # Dimension must match embedding model (1536 for ada-002)
        self.index = faiss.IndexFlatL2(1536)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add embeddings and their metadata to the FAISS index."""
        vecs = np.array(vectors, dtype='float32')
        self.index.add(vecs)
        self.metadata.extend(metadata)

    def query(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query the FAISS index and return top_k metadata records with distance score."""
        vec = np.array([vector], dtype='float32')
        distances, idxs = self.index.search(vec, top_k)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], idxs[0]):
            record = self.metadata[idx].copy()
            record['score'] = float(dist)
            results.append(record)
        return results