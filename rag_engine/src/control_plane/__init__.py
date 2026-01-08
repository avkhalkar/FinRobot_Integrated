"""
Control Plane - Manager Layer

Handles data lifecycle management:
- Existence checks (full onboarding if data doesn't exist)
- Staleness checks (incremental updates for stale components)
- Fetch, serialize, embed, and upsert operations

Invariant: Pinecone is always an exact mirror of disk (disk is source of truth).
"""

from .config import Jurisdiction, DataComponent, FRESHNESS_POLICIES, BASE_DATA_DIR
from .company_registry import get_company_info, register_company, CompanyInfo
from .freshness import check_component_freshness, FreshnessResult
from .manager import ControlPlaneManager, DataChecklist, ControlPlaneResult

__all__ = [
    "Jurisdiction",
    "DataComponent",
    "FRESHNESS_POLICIES",
    "BASE_DATA_DIR",
    "get_company_info",
    "register_company",
    "CompanyInfo",
    "check_component_freshness",
    "FreshnessResult",
    "ControlPlaneManager",
    "DataChecklist",
    "ControlPlaneResult",
]
