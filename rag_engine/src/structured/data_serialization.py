import pandas as pd
import os
import json

def seralize_paraquet(path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error reading {path} ; {e}")
        return []

    docs = []

    ticker = df["_meta_ticker"].iloc[0]
    report_type = df["_meta_report_type"].iloc[0]
    source = df["_meta_source"].iloc[0]
    fetched_at = df["_meta_fetched_at"].iloc[0]

    for index, row in df.iterrows():
        content_data = row.drop([c for c in row.index if c.startswith("_meta")])

        if "Date" in row and pd.notna(row["Date"]):
            date_str = str(row["Date"])
        else:
            date_str = "As of fetch time"

        text_parts = [
            f"Financial Report for {ticker} ({report_type}) on {date_str}:"
        ]

        for col, val in content_data.items():
            if pd.isna(val):
                continue

            if isinstance(val, (int, float)) and abs(val) > 1_000_000:
                val_str = f"{val:.0f}"
            else:
                val_str = str(val)

            text_parts.append(f"- {col}: {val_str}")

        docs.append({
            "id": f"{ticker}_{report_type}_{index}",
            "text": "\n".join(text_parts),
            "metadata": {
                "ticker": ticker,
                "report_type": report_type,
                "date": date_str,
                "source": source,
                "fetched_at": fetched_at
            }
        })

    return docs
