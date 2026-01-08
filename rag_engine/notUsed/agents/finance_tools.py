"""
langchain_financial_tools.py

Financial data extraction functions and LangChain tools.
"""

from typing import Optional, List, Dict, Any
import os
import time
import json
import requests
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Optional dependency
# -----------------------------
try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# LangChain tool decorator
# -----------------------------
try:
    from langchain.tools import tool
except Exception:
    def tool(*args, **kwargs):
        def deco(f):
            return f
        return deco

# -----------------------------
# Utility: Safe requests
# -----------------------------
class SafeRequester:
    def __init__(self, sleep_on_rate_limit=1.0, max_retries=3, backoff=2.0):
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self.max_retries = max_retries
        self.backoff = backoff

    def get(self, url, params=None, headers=None, timeout=30):
        attempt = 0
        while True:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=timeout)
                if resp.status_code in (429, 503):
                    attempt += 1
                    if attempt > self.max_retries:
                        resp.raise_for_status()
                    time.sleep(self.backoff ** attempt)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                attempt += 1
                if attempt > self.max_retries:
                    return None
                time.sleep(self.backoff ** attempt)

_safe_req = SafeRequester()

# -----------------------------
# Yahoo Finance
# -----------------------------
def fetch_stock_price(symbol: str) -> Dict[str, Any]:
    """Return latest stock price using Yahoo Finance."""
    if yf is None:
        raise ImportError("yfinance not installed")

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    info = getattr(ticker, "info", {}) or {}

    price = None
    timestamp = None
    if not hist.empty:
        row = hist.iloc[-1]
        price = float(row["Close"])
        timestamp = str(row.name)

    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": info.get("currency"),
        "timestamp": timestamp
    }

def fetch_historical_prices(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Return historical prices."""
    if yf is None:
        raise ImportError("yfinance not installed")
    return yf.Ticker(symbol).history(period=period)

# -----------------------------
# News API
# -----------------------------
def fetch_newsapi_articles(query: str, page_size: int = 10) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")
    if not api_key:
        return None

    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": page_size, "sortBy": "relevancy", "apiKey": api_key}
    headers = {"X-Api-Key": api_key}

    resp = _safe_req.get(url, params=params, headers=headers)
    return resp.json() if resp else None


# -----------------------------
# Local loaders
# -----------------------------
def load_csv_as_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_pdf_text(path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)

# -----------------------------
# LangChain tools
# -----------------------------
@tool("fetch_stock_price", return_direct=True)
def fetch_stock_price_tool(symbol: str) -> str:
    """
    Fetch latest stock price for a ticker symbol.
    The symbol may arrive wrapped in JSON or with whitespace.
    """
    try:
        # 🔑 CLEAN AGENT INPUT
        if isinstance(symbol, str):
            symbol = symbol.strip()

            # If agent passed JSON-like string, extract value
            if symbol.startswith("{"):
                parsed = json.loads(symbol)
                symbol = parsed.get("symbol") or parsed.get("SYMBOL")

        symbol = symbol.upper()

        data = fetch_stock_price(symbol)
        return json.dumps(data)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("fetch_historical_prices", return_direct=True)
def fetch_historical_prices_tool(symbol: str, period: str = "1y") -> str:
    """
    Fetch historical stock prices for a given ticker symbol over a specified period.
    Returns CSV-formatted text with Date and Close price.
    """
    try:
        if isinstance(symbol, str):
            symbol = symbol.strip()
            if symbol.startswith("{"):
                parsed = json.loads(symbol)
                symbol = parsed.get("symbol") or parsed.get("SYMBOL")

        symbol = symbol.upper()

        df = fetch_historical_prices(symbol, period)
        if df.empty:
            return json.dumps({"error": "No data returned"})

        out = df.reset_index()[["Date", "Close"]]
        return out.to_csv(index=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("fetch_news", return_direct=True)
def fetch_news_tool(query: str) -> str:
    """Fetch recent news articles related to a query."""
    data = fetch_newsapi_articles(query)
    return json.dumps(data or {"error": "News not available"})

def get_finance_tools():
    return [
        fetch_stock_price_tool,
        fetch_historical_prices_tool,
        fetch_news_tool
    ]
