import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timezone
from typing import Literal, cast
from pathlib import Path
import json

# Import for normalizing tickers
from src.control_plane.ticker_normalizer import normalize_for_yfinance


# Import serialization from same package
try:
    from .data_serialization import seralize_paraquet
except ImportError:
    from data_serialization import seralize_paraquet

# Base data directory (resolved from this file's location)
BASE_DIR = Path(__file__).resolve().parents[3] / "data"

def fetch_and_store_stock_data(
    ticker: str,
    report_type: Literal["price", "income_stmt", "balance_sheet", "cash_flow", "info"] = "price"
) -> str:

    print(f"Fetching {report_type} for {ticker}...")

    ticker = ticker.strip().upper()
    # Normalize ticker
    yf_ticker = normalize_for_yfinance(ticker)
    stock = yf.Ticker(yf_ticker)

    try:
        df = pd.DataFrame()

        if report_type == "price":
            df = stock.history(period="3mo", interval="1d")
            if df.empty:
                raise ValueError(f"No price data found for {ticker}")
            df.reset_index(inplace=True)

        elif report_type == "income_stmt":
            df = stock.financials.T
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date"}, inplace=True)

        elif report_type == "balance_sheet":
            df = stock.balance_sheet.T
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date"}, inplace=True)

        elif report_type == "cash_flow":
            df = stock.cashflow.T
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date"}, inplace=True)

        elif report_type == "info":
            info_dict = stock.info
            keys_to_keep = [
                "longName", "sector", "industry", "marketCap",
                "forwardPE", "dividendYield", "profitMargins", "totalRevenue"
            ]
            filtered_info = {k: [info_dict.get(k)] for k in keys_to_keep}
            df = pd.DataFrame(filtered_info)

        else:
            raise ValueError(f"Invalid report_type: {report_type}")

        fetched_at = datetime.now(timezone.utc)

        df["_meta_ticker"] = ticker
        df["_meta_report_type"] = report_type
        df["_meta_source"] = "yfinance"
        df["_meta_fetched_at"] = fetched_at.isoformat()
        df["_meta_data_version"] = "v1.0"

        out_dir = BASE_DIR / ticker / "structured"
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / f"{report_type}.parquet"
        df.to_parquet(parquet_path, index=False)

        docs = seralize_paraquet(str(parquet_path))

        json_path = out_dir / f"{report_type}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)

        print(f"Stored structured data for {ticker} ({report_type}) → {out_dir}")
        return str(parquet_path)

    except Exception as e:
        print(f"Error processing {ticker} ({report_type}): {e}")
        return ""

if __name__ == "__main__":
    watchlist = ["AAPL", "MSFT"]
    required_reports = ["price", "income_stmt", "balance_sheet", "cash_flow", "info"]

    for symbol in watchlist:
        for report in required_reports:
            fetch_and_store_stock_data(
                symbol,
                cast(Literal["price", "income_stmt", "balance_sheet", "cash_flow", "info"], report)
            )
