"""
Retrieval logic: embed query and fetch top_k chunks.
"""
from typing import List, Dict, Any
from .embedder import Embedder
from .vector_store import VectorStore
class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embedder.embed([query])[0]
        return self.store.query(q_vec, top_k)
