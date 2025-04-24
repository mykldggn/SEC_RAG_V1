"""
End‑to‑end orchestration: ingest → index → retrieve → extract → DataFrame
"""
import pandas as pd
import requests
from .config import CHUNK_SIZE, CHUNK_OVERLAP, FEATURES, EDGAR_USER_AGENT
from .edgar_client import EdgarClient
from .preprocess import html_to_text, split_sections
from .chunker import chunk_text
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .extractor import extract_fields

def run(query: str, cik: str) -> pd.DataFrame:
    #Fetch EDGAR filings
    client = EdgarClient()
    filings = client.fetch_filings(cik=cik, filing_types=["10-K", "10-Q", "8-K"])
    if not filings:
        raise RuntimeError(
            f"No 10-K/Q/8-K filings found for CIK={cik!r}. "
            "Check your CIK, EDGAR_USER_AGENT, or filing_types filter."
        )
    #Index chunk
    embedder = Embedder()
    store = VectorStore()
    #include UA
    headers = {"User-Agent": EDGAR_USER_AGENT}
    # debug: print the first few filing URLs
    if filings:
        print("🔗 Filing URLs:", [f["htmlUrl"] for f in filings][:3])
    for f in filings:
        resp = requests.get(f['htmlUrl'], headers=headers)
        resp.raise_for_status()
        html = resp.text
        text = html_to_text(html)
        for sec in split_sections(text):
            for chunk in chunk_text(sec, CHUNK_SIZE, CHUNK_OVERLAP):
                vec = embedder.embed([chunk])[0]
                store.add([vec], [{**f, 'text': chunk}])
    if not store.metadata:
        raise RuntimeError(
            "Fetched filings but no text sections got chunked/indexed. "
            "Verify that htmlUrl points at the raw filing (see pipeline.py patch)."
        )
    #Retrieve and extract
    retriever = Retriever(embedder, store)
    chunks = retriever.retrieve(query)
    records = extract_fields(chunks)
    #Build DataFrame
    df = pd.DataFrame(records)
    #Enforce dtypes
    for col, meta in FEATURES.items():
        if meta['type'] == 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col])
        elif meta['type'] == 'string':
            df[col] = df[col].astype(str)
    return df