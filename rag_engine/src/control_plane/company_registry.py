"""
Company Registry - Maps tickers to jurisdiction and identifiers.

Provides:
- Pre-populated registry of known companies
- Lookup by ticker symbol
- Dynamic registration for new companies
"""

from dataclasses import dataclass
from typing import Optional
from .config import Jurisdiction


@dataclass
class CompanyInfo:
    """Company identification information."""
    ticker: str
    jurisdiction: Jurisdiction
    cik: Optional[str] = None           # For US companies (SEC EDGAR)
    scrip_code: Optional[str] = None    # For Indian companies (BSE)


# Pre-populated registry of known companies
_COMPANY_REGISTRY: dict[str, CompanyInfo] = {
    # US Companies (SEC EDGAR)
    "AAPL": CompanyInfo(ticker="AAPL", jurisdiction=Jurisdiction.US, cik="0000320193"),
    "MSFT": CompanyInfo(ticker="MSFT", jurisdiction=Jurisdiction.US, cik="0000789019"),
    "GOOGL": CompanyInfo(ticker="GOOGL", jurisdiction=Jurisdiction.US, cik="0001652044"),
    "AMZN": CompanyInfo(ticker="AMZN", jurisdiction=Jurisdiction.US, cik="0001018724"),
    "META": CompanyInfo(ticker="META", jurisdiction=Jurisdiction.US, cik="0001326801"),
    "NVDA": CompanyInfo(ticker="NVDA", jurisdiction=Jurisdiction.US, cik="0001045810"),
    "TSLA": CompanyInfo(ticker="TSLA", jurisdiction=Jurisdiction.US, cik="0001318605"),
    "JPM": CompanyInfo(ticker="JPM", jurisdiction=Jurisdiction.US, cik="0000019617"),
    "V": CompanyInfo(ticker="V", jurisdiction=Jurisdiction.US, cik="0001403161"),
    "JNJ": CompanyInfo(ticker="JNJ", jurisdiction=Jurisdiction.US, cik="0000200406"),

    # Indian Companies (BSE)
    "TCS": CompanyInfo(ticker="TCS", jurisdiction=Jurisdiction.INDIA, scrip_code="532540"),
    "RELIANCE": CompanyInfo(ticker="RELIANCE", jurisdiction=Jurisdiction.INDIA, scrip_code="500325"),
    "INFY": CompanyInfo(ticker="INFY", jurisdiction=Jurisdiction.INDIA, scrip_code="500209"),
    "HDFCBANK": CompanyInfo(ticker="HDFCBANK", jurisdiction=Jurisdiction.INDIA, scrip_code="500180"),
    "ICICIBANK": CompanyInfo(ticker="ICICIBANK", jurisdiction=Jurisdiction.INDIA, scrip_code="532174"),
    "HINDUNILVR": CompanyInfo(ticker="HINDUNILVR", jurisdiction=Jurisdiction.INDIA, scrip_code="500696"),
    "ITC": CompanyInfo(ticker="ITC", jurisdiction=Jurisdiction.INDIA, scrip_code="500875"),
    "SBIN": CompanyInfo(ticker="SBIN", jurisdiction=Jurisdiction.INDIA, scrip_code="500112"),
    "BHARTIARTL": CompanyInfo(ticker="BHARTIARTL", jurisdiction=Jurisdiction.INDIA, scrip_code="532454"),
    "KOTAKBANK": CompanyInfo(ticker="KOTAKBANK", jurisdiction=Jurisdiction.INDIA, scrip_code="500247"),
}


def get_company_info(ticker: str) -> Optional[CompanyInfo]:
    """
    Look up company information by ticker symbol.

    Args:
        ticker: Stock ticker symbol (case-insensitive)

    Returns:
        CompanyInfo if found in registry, None otherwise
    """
    return _COMPANY_REGISTRY.get(ticker.upper())


def register_company(
    ticker: str,
    jurisdiction: Jurisdiction,
    cik: Optional[str] = None,
    scrip_code: Optional[str] = None
) -> CompanyInfo:
    """
    Register a new company in the registry.

    Args:
        ticker: Stock ticker symbol
        jurisdiction: US or INDIA
        cik: SEC CIK (required for US companies)
        scrip_code: BSE scrip code (required for Indian companies)

    Returns:
        The registered CompanyInfo

    Raises:
        ValueError: If required identifier is missing for jurisdiction
    """
    ticker = ticker.upper()

    if jurisdiction == Jurisdiction.US and not cik:
        raise ValueError("CIK is required for US companies")
    if jurisdiction == Jurisdiction.INDIA and not scrip_code:
        raise ValueError("Scrip code is required for Indian companies")

    info = CompanyInfo(
        ticker=ticker,
        jurisdiction=jurisdiction,
        cik=cik,
        scrip_code=scrip_code
    )

    _COMPANY_REGISTRY[ticker] = info
    return info


def resolve_company(
    ticker: str,
    cik: Optional[str] = None,
    scrip_code: Optional[str] = None
) -> Optional[CompanyInfo]:
    """
    Resolve company info from registry or provided identifiers.

    Priority:
    1. Look up in registry
    2. If not found, create from provided cik/scrip_code

    Args:
        ticker: Stock ticker symbol
        cik: Optional CIK override (determines US jurisdiction)
        scrip_code: Optional scrip code override (determines INDIA jurisdiction)

    Returns:
        CompanyInfo if resolvable, None if unknown and no identifiers provided
    """
    ticker = ticker.upper()

    # First check registry
    info = get_company_info(ticker)

    if info:
        # Override with provided values if any
        if cik:
            info = CompanyInfo(
                ticker=info.ticker,
                jurisdiction=Jurisdiction.US,
                cik=cik,
                scrip_code=info.scrip_code
            )
        if scrip_code:
            info = CompanyInfo(
                ticker=info.ticker,
                jurisdiction=Jurisdiction.INDIA,
                cik=info.cik,
                scrip_code=scrip_code
            )
        return info

    # Not in registry - try to create from provided identifiers
    if cik:
        return CompanyInfo(
            ticker=ticker,
            jurisdiction=Jurisdiction.US,
            cik=cik,
            scrip_code=None
        )
    elif scrip_code:
        return CompanyInfo(
            ticker=ticker,
            jurisdiction=Jurisdiction.INDIA,
            cik=None,
            scrip_code=scrip_code
        )

    return None


def list_registered_tickers() -> list[str]:
    """Return list of all registered ticker symbols."""
    return list(_COMPANY_REGISTRY.keys())
