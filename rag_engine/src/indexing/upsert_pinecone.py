from dotenv import load_dotenv

load_dotenv()
import os
import sys
import json

from pinecone import Pinecone


from src.indexing.chunking import chunk_document
from src.embeddings.embedding_provider import embed_texts




INDEX_NAME = "financial-rag"
BATCH_SIZE = 32



api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    print("ERROR: PINECONE_API_KEY not found in environment.")
    sys.exit(1)

pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

def valid_text(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 50

def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


#  Bulk upsert (S + U data)

#  S
def index_unstructured(ticker: str, base_path: str):
    path = os.path.join(base_path, "unstructured", "data.json")
    if not os.path.exists(path):
        print(f"Skipping Unstructured: File not found at {path}")
        return

    print(f"Reading unstructured data for {ticker}...")
    with open(path, "r") as f:
        doc = json.load(f)

    print(f"Chunking unstructured document...")
    chunks = chunk_document(doc)
    
    ids, texts, metas = [], [], []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not valid_text(text):
            continue

        ids.append(chunk["id"])
        texts.append(text)
        metas.append({
            "ticker": ticker,
            "text": text,
            "source": chunk.get("source"),
            "data_category": "narrative"
        })

    if not texts:
        print("No valid text found in unstructured data.")
        return

    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = embed_texts(texts)

    print(f"Upserting {len(vectors)} vectors to Pinecone (Namespace: {ticker})")
    for batch in batched(list(zip(ids, vectors, metas)), BATCH_SIZE):
        index.upsert(vectors=batch, namespace=ticker)
    print("Unstructured indexing complete.")


#  U
def index_narrated_financials(ticker: str, base_path: str):
    struct_dir = os.path.join(base_path, "structured")
    if not os.path.exists(struct_dir):
        print(f"Skipping Structured: Directory not found at {struct_dir}")
        return

    print(f"Scanning structured directory: {struct_dir}")
    grouped = {}
    for fname in os.listdir(struct_dir):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(struct_dir, fname), "r") as f:
            records = json.load(f)

        for record in records:
            meta = record["metadata"]
            key = (meta["report_type"], meta["date"])
            grouped.setdefault(key, []).append(record["text"])

    ids, texts, metas = [], [], []
    for (report_type, date), parts in grouped.items():
        combined_text = "\n".join(parts)
        if not valid_text(combined_text):
            continue

        doc_id = f"{ticker}_{report_type}_{date}"
        ids.append(doc_id)
        texts.append(combined_text)
        metas.append({
            "ticker": ticker,
            "text": combined_text,
            "report_type": report_type,
            "fiscal_date": date,
            "data_category": "narrated_numeric"
        })

    if not texts:
        print("No valid structured records found.")
        return

    print(f"Generating embeddings for {len(texts)} financial summaries...")
    vectors = embed_texts(texts)

    print(f"Upserting {len(vectors)} financial vectors to Pinecone...")
    for batch in batched(list(zip(ids, vectors, metas)), BATCH_SIZE):
        index.upsert(vectors=batch, namespace=ticker)
    print("Structured indexing complete.")



# Base data directory (resolved from this file's location)
from pathlib import Path
BASE_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


#  S + U
def index_all_data(ticker: str, base_path: str = None):
    """
    Index all data (structured + unstructured) for a ticker.

    Args:
        ticker: Stock ticker symbol
        base_path: Optional base path override (default: project_root/data/TICKER)
    """
    if base_path is None:
        base_path = str(BASE_DATA_DIR / ticker)

    print(f"\nStarting full indexing for: {ticker}")
    print("=" * 40)

    index_unstructured(ticker, base_path)
    print("-" * 20)
    index_narrated_financials(ticker, base_path)

    print("=" * 40)
    print(f"Finished all tasks for {ticker}\n")

#  Incremental upert

def get_index():
    return index

def upsert_to_namespace(ids, vectors, metas, ticker):
    """Upsert vectors to a specific namespace (ticker)."""
    payload = list(zip(ids, vectors, metas))
    for batch in batched(payload, BATCH_SIZE):
        index.upsert(vectors=batch, namespace=ticker)




# Test/Main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.indexing.upsert_pinecone <TICKER>")
        sys.exit(1)

    ticker_input = sys.argv[1].upper()
    index_all_data(ticker_input)

