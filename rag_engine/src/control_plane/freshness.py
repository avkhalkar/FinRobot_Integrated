"""
Freshness Checker - Determines if data components are stale.

Reads fetched_at timestamps from disk and compares against freshness policies.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import json
import os

from .config import Jurisdiction, FRESHNESS_POLICIES, DataComponent


@dataclass
class FreshnessResult:
    """Result of a freshness check for a data component."""
    component: str
    exists: bool
    is_fresh: bool
    fetched_at: Optional[datetime] = None
    age: Optional[timedelta] = None
    policy: Optional[timedelta] = None

    def __str__(self) -> str:
        if not self.exists:
            return f"{self.component}: NOT FOUND"
        age_str = str(self.age).split('.')[0] if self.age else "unknown"
        status = "FRESH" if self.is_fresh else "STALE"
        return f"{self.component}: {status} (age: {age_str})"


def get_fetched_at_from_parquet(parquet_path: Path) -> Optional[datetime]:
    """
    Read _meta_fetched_at from a structured parquet file.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        datetime of when data was fetched, or None if not found
    """
    try:
        df = pd.read_parquet(parquet_path)
        if "_meta_fetched_at" not in df.columns:
            return None

        fetched_at_str = df["_meta_fetched_at"].iloc[0]
        return datetime.fromisoformat(fetched_at_str)
    except Exception:
        return None


def get_fetched_at_from_json(json_path: Path) -> Optional[datetime]:
    """
    Read fetched_at from a JSON file (unstructured data).

    Args:
        json_path: Path to the JSON file

    Returns:
        datetime of when data was fetched, or None if not found
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fetched_at_str = data.get("fetched_at")
        if not fetched_at_str:
            return None

        return datetime.fromisoformat(fetched_at_str)
    except Exception:
        return None


def get_latest_pdf_date(raw_dir: Path) -> Optional[datetime]:
    """
    Get the most recent date from BSE PDF filenames.

    PDF files are named: YYYY-MM-DD_subject.pdf or YYYYMMDD_subject.pdf

    Args:
        raw_dir: Directory containing PDF files

    Returns:
        datetime of the most recent PDF, or None if no valid PDFs found
    """
    if not raw_dir.exists():
        return None

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        return None

    latest_date: Optional[datetime] = None

    for pdf_file in pdf_files:
        filename = pdf_file.stem  # e.g., "2024-06-15_Annual_Report"
        date_part = filename.split("_")[0]

        # Try different date formats
        for fmt in ["%Y-%m-%d", "%Y%m%d"]:
            try:
                file_date = datetime.strptime(date_part, fmt)
                file_date = file_date.replace(tzinfo=timezone.utc)
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                break
            except ValueError:
                continue

    # Fallback to file modification time if no valid dates parsed
    if latest_date is None and pdf_files:
        try:
            mod_times = [os.path.getmtime(f) for f in pdf_files]
            latest_date = datetime.fromtimestamp(max(mod_times), tz=timezone.utc)
        except Exception:
            pass

    return latest_date


def check_structured_freshness(
    ticker: str,
    component: str,
    base_dir: Path
) -> FreshnessResult:
    """
    Check freshness of a structured data component.

    Args:
        ticker: Stock ticker symbol
        component: Component name (price, income_stmt, etc.)
        base_dir: Base data directory

    Returns:
        FreshnessResult with existence and freshness status
    """
    parquet_path = base_dir / ticker / "structured" / f"{component}.parquet"
    policy = FRESHNESS_POLICIES.get(component, timedelta(hours=24))

    if not parquet_path.exists():
        return FreshnessResult(
            component=component,
            exists=False,
            is_fresh=False,
            policy=policy
        )

    fetched_at = get_fetched_at_from_parquet(parquet_path)

    if fetched_at is None:
        return FreshnessResult(
            component=component,
            exists=True,
            is_fresh=False,
            policy=policy
        )

    # Ensure timezone-aware comparison
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age = now - fetched_at
    is_fresh = age < policy

    return FreshnessResult(
        component=component,
        exists=True,
        is_fresh=is_fresh,
        fetched_at=fetched_at,
        age=age,
        policy=policy
    )


def check_unstructured_freshness(
    ticker: str,
    jurisdiction: Jurisdiction,
    base_dir: Path
) -> FreshnessResult:
    """
    Check freshness of unstructured data.

    For US: Checks data.json fetched_at
    For India: Checks latest PDF file date

    Args:
        ticker: Stock ticker symbol
        jurisdiction: US or INDIA
        base_dir: Base data directory

    Returns:
        FreshnessResult with existence and freshness status
    """
    policy = FRESHNESS_POLICIES.get("unstructured", timedelta(days=365))

    if jurisdiction == Jurisdiction.US:
        json_path = base_dir / ticker / "unstructured" / "data.json"

        if not json_path.exists():
            return FreshnessResult(
                component="unstructured",
                exists=False,
                is_fresh=False,
                policy=policy
            )

        fetched_at = get_fetched_at_from_json(json_path)

    else:  # INDIA
        raw_dir = base_dir / ticker / "unstructured" / "raw"

        if not raw_dir.exists() or not any(raw_dir.glob("*.pdf")):
            return FreshnessResult(
                component="unstructured",
                exists=False,
                is_fresh=False,
                policy=policy
            )

        fetched_at = get_latest_pdf_date(raw_dir)

    if fetched_at is None:
        return FreshnessResult(
            component="unstructured",
            exists=True,
            is_fresh=False,
            policy=policy
        )

    # Ensure timezone-aware comparison
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age = now - fetched_at
    is_fresh = age < policy

    return FreshnessResult(
        component="unstructured",
        exists=True,
        is_fresh=is_fresh,
        fetched_at=fetched_at,
        age=age,
        policy=policy
    )


def check_component_freshness(
    ticker: str,
    component: DataComponent,
    jurisdiction: Jurisdiction,
    base_dir: Path
) -> FreshnessResult:
    """
    Check freshness of any data component.

    Args:
        ticker: Stock ticker symbol
        component: Component name
        jurisdiction: US or INDIA (needed for unstructured check)
        base_dir: Base data directory

    Returns:
        FreshnessResult with existence and freshness status
    """
    if component == "unstructured":
        return check_unstructured_freshness(ticker, jurisdiction, base_dir)
    else:
        return check_structured_freshness(ticker, component, base_dir)


def check_all_freshness(
    ticker: str,
    jurisdiction: Jurisdiction,
    base_dir: Path,
    components: Optional[list[str]] = None
) -> dict[str, FreshnessResult]:
    """
    Check freshness of all specified components for a ticker.

    Args:
        ticker: Stock ticker symbol
        jurisdiction: US or INDIA
        base_dir: Base data directory
        components: List of components to check (default: all)

    Returns:
        Dict mapping component names to FreshnessResult
    """
    from .config import STRUCTURED_COMPONENTS

    if components is None:
        components = STRUCTURED_COMPONENTS + ["unstructured"]

    results = {}
    for component in components:
        results[component] = check_component_freshness(
            ticker=ticker,
            component=component,
            jurisdiction=jurisdiction,
            base_dir=base_dir
        )

    return results


def ticker_folder_exists(ticker: str, base_dir: Path) -> bool:
    """Check if the ticker's data folder exists."""
    return (base_dir / ticker).exists()
