"""
Configuration for the Control Plane.

Defines:
- Jurisdiction enum (US, INDIA)
- Data component types
- Freshness policies (max age before data is considered stale)
- Base paths and constants
"""

from datetime import timedelta
from pathlib import Path
from enum import Enum
from typing import Literal


class Jurisdiction(Enum):
    """Supported company jurisdictions."""
    US = "US"
    INDIA = "INDIA"


# Data component types that can be fetched/indexed
DataComponent = Literal[
    "price",
    "income_stmt",
    "balance_sheet",
    "cash_flow",
    "info",
    "unstructured"
]

# All structured component types
STRUCTURED_COMPONENTS: list[str] = [
    "price",
    "income_stmt",
    "balance_sheet",
    "cash_flow",
    "info"
]

# Freshness policies: max age before component is considered stale
FRESHNESS_POLICIES: dict[str, timedelta] = {
    "price": timedelta(hours=24),           # Daily price updates
    "income_stmt": timedelta(days=90),      # Quarterly financials
    "balance_sheet": timedelta(days=90),
    "cash_flow": timedelta(days=90),
    "info": timedelta(days=7),              # Company info weekly
    "unstructured": timedelta(days=365),    # 10-K filings yearly
}

# Base data directory (resolves to project_root/data)
BASE_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Pinecone configuration
PINECONE_INDEX_NAME = "financial-rag"
EMBEDDING_BATCH_SIZE = 32
