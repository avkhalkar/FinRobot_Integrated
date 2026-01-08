import os
import json
from chunking import chunk_document 

BASE_DIR = "../../../data"

def process_all_unstructured_data():
    """
    Iterates through all company folders to find and chunk unstructured data.json.
    """
    
    tickers = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    all_chunks = []

    for ticker in tickers:
     
        file_path = os.path.join(BASE_DIR, ticker, "unstructured", "data.json")
        
        if not os.path.exists(file_path):
            print(f"Skipping {ticker}: No unstructured data.json found.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            raw_doc = json.load(f)

       
        doc_for_chunking = {
            "ticker": raw_doc["company"], 
            "text": raw_doc["text"],
            "source": raw_doc["source"],
            "jurisdiction": raw_doc["jurisdiction"],
            "fetched_at": raw_doc["fetched_at"]
        }

 
        chunks = chunk_document(doc_for_chunking)
        print(f"Generated {len(chunks)} chunks for {ticker}")
        all_chunks.extend(chunks)

    
    return all_chunks

if __name__ == "__main__":
    processed_chunks = process_all_unstructured_data()
    print(f"Total chunks processed: {len(processed_chunks)}")