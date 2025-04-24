"""
Configuration constants and schema for the SEC‑filing RAG pipeline.
"""
from dotenv import load_dotenv, find_dotenv
# Load environment variables from .env
load_dotenv(find_dotenv())
import os
import openai

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")
openai.api_key = OPENAI_API_KEY

# OpenAI models
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL",      "gpt-4o-mini")

# Chunking parameters
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    1000))  # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))   # token overlap

# Vector store settings (FAISS only)
VECTOR_DB = "faiss"  # local in‑memory indexing

# EDGAR settings
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "your-email@example.com")

# Schema for DataFrame columns as per PM requirements
FEATURES = {
    "Date": {
        "description": "The date when the update was made",
        "type": "datetime64[ns]",
        "required": True
    },
    "Ticker": {
        "description": "The ticker of stock being updated on Date",
        "type": "string",
        "required": True
    },
    "Direction": {
        "description": "Price target 'raised' or 'lowered' on Date",
        "type": "enum",
        "enum": ["raised", "lowered"],
        "required": True
    }
}