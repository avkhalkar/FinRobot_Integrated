import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Load Env
load_dotenv(project_root / ".env")

# Import pinecone_client
from retrieval.pinecone_client import smart_retrieve

def test_retrieval():
    print("Testing Integrated Retrieval...")
    query = "What represents the revenue of Apple?"
    print(f"Query: {query}")
    
    try:
        chunks = smart_retrieve(query)
        print(f"\nRetrieved {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks[:2]):
            print(f"-- Chunk {i+1} --")
            print(f"ID: {chunk.id}")
            print(f"Text: {chunk.text[:100]}...")
            print(f"Score: {chunk.score}")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_retrieval()
