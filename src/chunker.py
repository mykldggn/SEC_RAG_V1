"""
Overlap‐based text chunker using GPT‑2 tokenizer.
"""
from typing import List
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += size - overlap
    return chunks