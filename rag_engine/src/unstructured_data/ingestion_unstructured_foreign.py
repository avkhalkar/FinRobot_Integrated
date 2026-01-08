"""

** Please Read Once before Modifying or Using ** ~DhanushHN10
THIS SCRIPT IS AN INGESTION WORKER FOR *US / SEC-LISTED COMPANIES ONLY*.

It MUST be called by an upstream system that has already:
1. Resolved the user prompt into a concrete company entity
2. Determined the jurisdiction is the United States
3. Supplied a VALID SEC CIK for that company

This script DOES NOT:
- Guess company identity
- Guess country
- Decide which ingestion pipeline to use
- Handle Indian companies (BSE/NSE)
- Handle user prompts directly

If a non-US company or an invalid CIK is passed here,
the script is expected to FAIL FAST to prevent RAG corruption.

Output contract (MANDATORY, DO NOT CHANGE):
data/
  └── {COMPANY}/
      ├── structured/
      │   └── data.json
      └── unstructured/
          └── data.json

The output file contains BOTH content and metadata.
This script always fetches the latest available 10-K at runtime.
"""

import os
import json
import requests
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from bs4 import XMLParsedAsHTMLWarning
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Base data directory (resolved from this file's location)
BASE_DIR = Path(__file__).resolve().parents[3] / "data"

# SEC EDGAR requires a properly formatted User-Agent with contact info
# See: https://www.sec.gov/os/accessing-edgar-data
HEADERS = {
    "User-Agent": "FinancialRAGBot/1.0 (aiclub@iitdh.ac.in)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

MAX_OUTPUT_CHARS = 80_000


def get_latest_10k_metadata(cik: str, retries: int = 3) -> dict:
    """
    Fetch metadata for the latest 10-K filing from SEC EDGAR.

    Args:
        cik: Central Index Key (with or without leading zeros)
        retries: Number of retry attempts

    Returns:
        Dict with accession and filing_date
    """
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()

            filings = r.json()["filings"]["recent"]

            for form, acc, date in zip(
                filings["form"],
                filings["accessionNumber"],
                filings["filingDate"],
            ):
                if form == "10-K":
                    return {
                        "accession": acc.replace("-", ""),
                        "filing_date": date,
                    }

            raise RuntimeError("No 10-K filing found.")

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to fetch SEC metadata after {retries} attempts: {e}")

    raise RuntimeError("No 10-K filing found.")


def find_real_10k_html(cik: str, accession: str, retries: int = 3) -> str:
    """Find the actual 10-K HTML file from the filing index."""
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession}/index.json"
    )

    # Different headers for www.sec.gov
    sec_headers = {
        "User-Agent": "FinancialRAGBot/1.0 (aiclub@iitdh.ac.in)",
        "Accept-Encoding": "gzip, deflate",
    }

    for attempt in range(retries):
        try:
            r = requests.get(index_url, headers=sec_headers, timeout=30)
            r.raise_for_status()

            candidates = []

            for f in r.json()["directory"]["item"]:
                name = f.get("name", "").lower()

                if not name.endswith((".htm", ".html")):
                    continue
                if "ix" in name or "xbrl" in name:
                    continue

                try:
                    size = int(f.get("size", 0))
                except (TypeError, ValueError):
                    size = 0

                candidates.append((size, name))

            if not candidates:
                raise RuntimeError("No narrative HTML found.")

            _, best = max(candidates)

            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession}/{best}"
            )

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to find 10-K HTML after {retries} attempts: {e}")

    raise RuntimeError("No narrative HTML found.")


def normalize_html_to_text(html: str) -> str:
    """Convert HTML to clean text using BeautifulSoup."""
    html = re.sub(r"</?ix:[^>]+>", " ", html, flags=re.I)

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text).strip()

    start = re.search(
        r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION",
        text,
        re.I,
    )
    if start:
        text = text[start.start():]

    if len(text) < 100_000:
        raise RuntimeError("Not a full 10-K narrative.")

    return text


def extract_high_signal_text(text: str) -> str:
    """Extract key sections from 10-K text."""
    def between(start, end):
        s = re.search(start, text, re.I | re.S)
        e = re.search(end, text, re.I | re.S)
        if s and e and e.start() > s.end():
            return text[s.start():e.start()]
        return None

    sections = []

    business = between(
        r"Item\s+1\b.*?Business",
        r"Item\s+1A\b",
    )

    risk = between(
        r"Item\s+1A\b",
        r"Item\s+1B\b",
    )

    mdna = between(
        r"Item\s+7\b.*?Management",
        r"Item\s+7A\b",
    )

    market_risk = between(
        r"Item\s+7A\b",
        r"Item\s+8\b",
    )

    for sec in (business, risk, mdna, market_risk):
        if sec and len(sec) > 5_000:
            sections.append(sec.strip())

    if not sections:
        return text[:MAX_OUTPUT_CHARS]

    final = "\n\n".join(sections)

    if len(final) > MAX_OUTPUT_CHARS:
        final = final[:MAX_OUTPUT_CHARS]

    return final


def ingest_sec_unstructured(*, ticker: str, cik: str) -> dict:
    """
    Ingest SEC 10-K filing for a US company.

    Args:
        ticker: Stock ticker symbol
        cik: SEC Central Index Key

    Returns:
        Dict with company info, filing metadata, and extracted text

    Raises:
        ValueError: If CIK is invalid
        RuntimeError: If no 10-K found or fetch fails
    """
    if not cik.isdigit():
        raise ValueError("Invalid CIK.")

    ticker = ticker.upper()
    print(f"[SEC] Fetching 10-K for {ticker} (CIK: {cik})...")

    meta = get_latest_10k_metadata(cik)
    print(f"[SEC] Found 10-K filed on {meta['filing_date']}, fetching content...")

    filing_url = find_real_10k_html(cik, meta["accession"])

    # Different headers for www.sec.gov
    sec_headers = {
        "User-Agent": "FinancialRAGBot/1.0 (aiclub@iitdh.ac.in)",
        "Accept-Encoding": "gzip, deflate",
    }
    html = requests.get(filing_url, headers=sec_headers, timeout=60).text
    full_text = normalize_html_to_text(html)
    signal_text = extract_high_signal_text(full_text)

    record = {
        "company": ticker,
        "ticker": ticker,
        "jurisdiction": "US",
        "source": "SEC EDGAR",
        "filing_type": "10-K",
        "filing_date": meta["filing_date"],
        "accession": meta["accession"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data_version": "v5.0",
        "text": signal_text,
    }

    # Save to disk using pathlib
    out_dir = BASE_DIR / ticker / "unstructured"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"[SEC] Saved {len(signal_text)} chars to {out_path}")
    return record


if __name__ == "__main__":
    res = ingest_sec_unstructured(
        ticker="MSFT",
        cik="0000789019",
    )
    print("Stored chars:", len(res["text"]))
