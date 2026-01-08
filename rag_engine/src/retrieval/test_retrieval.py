from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone

from src.embeddings.embedding_provider import embed_query


INDEX_NAME = "financial-rag"
TICKER = "MSFT"

QUERIES = ["What are the recent risk factors mentioned in the 10-K and how do they relate to Microsoft's total assets and net income in 2024?", "What was the Net Income in 2024?"]


def run_test(QUERY: str):
    print(f"\nStarting Retrieval Test for: {TICKER}")
    print("="*50)

   
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not found.")
        return

    print(f"Connecting to Pinecone index: {INDEX_NAME}...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

 
    print(f"Embedding query: '{QUERY}'...")
    try:
        qvec = embed_query(QUERY)
        print(f"Query embedded (Dimensions: {len(qvec)})")
    except Exception as e:
        print(f"Embedding Error: {e}")
        return

    print(f"Searching namespace '{TICKER}' for top 5 matches...")
    try:
        res = index.query(
            vector=qvec,
            top_k=5,
            namespace=TICKER,
            include_metadata=True
        )
    except Exception as e:
        print(f"Pinecone Query Error: {e}")
        return


    matches = res.get("matches", [])
    print(f"📊 Found {len(matches)} relevant matches.\n")

    for i, match in enumerate(matches, 1):
        metadata = match.get("metadata", {})
        category = metadata.get("data_category", "unknown")
        score = match.get("score", 0.0)
        
        print(f"--- Result {i} [Score: {score:.4f}] ---")
        print(f"Category: {category}")
        
        if category == "narrative":
            print(f"Source: {metadata.get('source')} | Section: {metadata.get('section')}")
        else:
            print(f"Report: {metadata.get('report_type')} | Date: {metadata.get('fiscal_date')}")
        
        
        text = metadata.get("text", "No text available")
        print(f"Content: {text[:200]}...")
        print("-" * 30 + "\n")

    print("="*50)
    print("Test Complete.")

if __name__ == "__main__":
    for QUERY in QUERIES:
        run_test(QUERY)