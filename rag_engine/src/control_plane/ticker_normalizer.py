# RAG/src/control_plane/ticker_normalizer.py

from src.control_plane.company_registry import get_company_info
from src.control_plane.config import Jurisdiction


def normalize_for_yfinance(ticker: str) -> str:
    """
    Convert logical ticker to Yahoo Finance compatible ticker.
    """
    ticker = ticker.strip().upper()

    info = get_company_info(ticker)

    if info and info.jurisdiction == Jurisdiction.INDIA:
        return f"{ticker}.NS"

    return ticker