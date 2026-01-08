"""
BSE India Ingestion Worker

Fetches corporate filings from BSE (Bombay Stock Exchange) for Indian companies.
Downloads PDF attachments for further processing.

Note: BSE has rate limiting in place. This script includes delays between requests.
"""

import os
import requests
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Base data directory (resolved from this file's location)
BASE_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data"

CHUNK_SIZE_DAYS = 90
TOTAL_HISTORY_DAYS = 365

# BSE API endpoints - updated for 2024/2025
BSE_BASE_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
BSE_PDF_URL_LIVE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"
BSE_PDF_URL_HIST = "https://www.bseindia.com/xml-data/corpfiling/AttachHis/"

# Headers that mimic a browser to avoid blocking
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    "Connection": "keep-alive",
}



def ensure_dirs(ticker: str) -> Path:
    """
    Create and return the directory path for storing PDFs.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Path to the raw PDFs directory
    """
    path = BASE_OUTPUT_DIR / ticker / "unstructured" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_date_chunks(days_back: int):
  
    chunks = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    current_start = start_date
    while current_start < end_date:
        current_end = current_start + timedelta(days=CHUNK_SIZE_DAYS)
        if current_end > end_date:
            current_end = end_date

        chunks.append((
            current_start.strftime("%Y%m%d"),
            current_end.strftime("%Y%m%d"),
        ))

        current_start = current_end + timedelta(days=1)

    return chunks



def fetch_bse_metadata_chunk(scrip_code: str, date_from: str, date_to: str, retries: int = 3) -> list:
    """
    Fetch filing metadata from BSE for a date range.

    Args:
        scrip_code: BSE scrip code
        date_from: Start date (YYYYMMDD)
        date_to: End date (YYYYMMDD)
        retries: Number of retry attempts

    Returns:
        List of filing metadata dicts
    """
    params = {
        "pageno": "1",
        "strCat": "-1",
        "strPrevDate": date_from,
        "strScrip": scrip_code,
        "strSearch": "P",
        "strToDate": date_to,
        "strType": "C",
        "subcategory": "-1"
    }

    for attempt in range(retries):
        try:
            # Create a session to handle cookies
            session = requests.Session()

            # First, visit the main page to get cookies
            session.get("https://www.bseindia.com/", headers=HEADERS, timeout=10)

            # Then make the API call
            r = session.get(
                BSE_BASE_URL,
                headers=HEADERS,
                params=params,
                timeout=30,
            )

            if r.status_code == 301 or r.status_code == 302:
                # Handle redirect - BSE sometimes redirects
                print(f"  Redirect detected, following...")
                r = session.get(r.headers.get("Location", ""), headers=HEADERS, timeout=30)

            r.raise_for_status()

            # Try to parse JSON
            try:
                data = r.json()
                return data.get("Table") or data.get("Table1") or []
            except json.JSONDecodeError:
                # Sometimes BSE returns HTML instead of JSON
                if "showinterest" in r.text.lower():
                    print(f"  BSE returned login page, retrying...")
                    time.sleep(2)
                    continue
                return []

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Error fetching {date_from} → {date_to}: {e}")
                return []

    return []


def process_company(ticker: str, scrip_code: str) -> dict:
    """
    Process a company: fetch metadata and download PDFs.

    Args:
        ticker: Stock ticker symbol
        scrip_code: BSE scrip code

    Returns:
        Dict with processing summary
    """
    ticker = ticker.upper()
    print(f"\n[BSE] Processing {ticker} (scrip: {scrip_code})")
    save_dir = ensure_dirs(ticker)

    date_chunks = get_date_chunks(TOTAL_HISTORY_DAYS)
    all_pdfs = []

    for start, end in date_chunks:
        print(f"  Querying BSE from {start} to {end}")
        rows = fetch_bse_metadata_chunk(scrip_code, start, end)

        for row in rows:
            fname = row.get("ATTACHMENTNAME")
            subject = row.get("NEWSSUB") or "Document"
            date = row.get("NEWS_DT") or "UnknownDate"
            is_old = row.get("OLD") == 1

            if fname and fname.lower().endswith(".pdf"):
                # Use historical URL for old filings, live URL for recent ones
                base_url = BSE_PDF_URL_HIST if is_old else BSE_PDF_URL_LIVE
                all_pdfs.append({
                    "url": base_url + fname,
                    "subject": subject,
                    "date": date,
                    "filename": fname,
                })

        time.sleep(1.5)  # Rate limiting

    print(f"[BSE] Found {len(all_pdfs)} PDF filings")

    downloaded = 0
    skipped = 0
    failed = 0

    for doc in all_pdfs:
        # Create safe filename
        safe_subject = "".join(
            c if c.isalnum() else "_"
            for c in doc["subject"][:60]
        )
        safe_date = doc["date"].split("T")[0].replace("-", "")
        filename = f"{safe_date}_{safe_subject}.pdf"
        filepath = save_dir / filename

        if filepath.exists():
            skipped += 1
            continue

        try:
            print(f"  Downloading: {filename[:50]}...")
            r = requests.get(doc["url"], headers=HEADERS, timeout=30)
            r.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(r.content)
            downloaded += 1
            time.sleep(0.5)  # Rate limiting between downloads

        except Exception as e:
            print(f"  [!] Failed: {filename[:30]}... - {e}")
            failed += 1

    # Save metadata file for freshness tracking
    metadata = {
        "ticker": ticker,
        "scrip_code": scrip_code,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_filings": len(all_pdfs),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }

    metadata_path = save_dir.parent / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[BSE] Complete: {downloaded} new, {skipped} existing, {failed} failed")
    return metadata



if __name__ == "__main__":
    targets = [
        {"ticker": "TCS", "scrip": "532540"},
      
    ]

    for t in targets:
        process_company(t["ticker"], t["scrip"])
