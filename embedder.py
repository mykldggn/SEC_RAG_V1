"""
Wrapper around OpenAI Embeddings API using the global OpenAI client.
"""
import openai
from typing import List
from .config import OPENAI_EMBEDDING_MODEL
class Embedder:
    def __init__(self, model: str = None):
        self.model = model or OPENAI_EMBEDDING_MODEL
    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = openai.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in resp.data]