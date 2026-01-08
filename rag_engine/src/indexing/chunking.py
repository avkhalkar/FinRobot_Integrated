import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS_PER_DOC = 50 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def stable_chunk_id(ticker: str, idx: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{ticker}_{idx}_{h}"

def chunk_document(doc: dict) -> list[dict]:
    raw_text = doc["text"]
    ticker = doc.get("ticker") or doc.get("company")
    if not ticker:
        raise KeyError("Document must contain either 'ticker' or 'company' key")
    splits = splitter.split_text(raw_text)

    chunks = []
    for i, chunk in enumerate(splits[:MAX_CHUNKS_PER_DOC]):
        chunks.append({
            "id": stable_chunk_id(ticker, i, chunk),
            "ticker": ticker,
            "text": chunk,
            "source": doc["source"],
            "jurisdiction": doc["jurisdiction"],
            "fetched_at": doc["fetched_at"]
        })

    return chunks
