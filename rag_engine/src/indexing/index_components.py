import json
from pathlib import Path
from src.indexing.chunking import chunk_document
from src.indexing.upsert_pinecone import (valid_text, upsert_to_namespace)
from src.embeddings.embedding_provider import embed_texts

DATA_DIR = Path(__file__).resolve().parents[3] / "data"




def index_component(ticker: str, report_type:str):
    ticker= ticker.upper()
    ids, texts, metas =[],[],[]

    if report_type =="unstructured":
        path = DATA_DIR/ticker/"unstructured"/"data.json"
        if not path.exists():
            print(f"Skipping Unstructured: File not found at {path}")
            return
        
        with open(path) as f:
            bundle = json.load(f)

        for doc in bundle.get("documents",[]):
            raw_text=doc.get("text","")
            if not valid_text(raw_text):
                continue

            chunks = chunk_document(raw_text)

            for i,chunk in enumerate(chunks):
                if not valid_text(chunk):
                    continue

                ids.append(f"{doc['id']}_chunk_{i}")
                texts.append(chunk)
                metas.append({
                    **doc["metadata"],
                    "ticker": ticker,
                    "text": chunk,
                })
        
    else :

        struct_dir = DATA_DIR/ticker/"structured"/f"{report_type}.json"

        if not struct_dir.exists():
            print(f"Skipping Structured: File not found at {struct_dir}")
            return
        
        with open(struct_dir) as f:
            records=json.load(f)


        for record in records:
            raw_text=record.get("text","")
            if not valid_text(raw_text):
                continue

            ids.append(record["id"])
            texts.append(raw_text)
            metas.append({
                **record["metadata"],
                "ticker": ticker,
                "text": raw_text,
            })


    if not texts:
        print(f"No valid text found in {report_type} data for {ticker}.")
        return
    

    vectors=embed_texts(texts)
    upsert_to_namespace(ids,vectors,metas,ticker)
    print(f"{report_type.capitalize()} indexing complete for {ticker}.")



