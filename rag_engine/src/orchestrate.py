"""
Unified Orchestrator - Entry point for the financial analysis pipeline.

Combines Control Plane (data management) and Inference Plane (retrieval).

Usage:
    from src.orchestrate import orchestrate

    # Basic usage
    result = orchestrate("AAPL", "What are Apple's risk factors?")

    # With custom checklist
    result = orchestrate(
        "MSFT",
        "Revenue trends",
        checklist=DataChecklist(structured=["price", "income_stmt"], unstructured=True)
    )

    # New company not in registry
    result = orchestrate("NVDA", "GPU market analysis", cik="0001045810")

    # Force refresh
    result = orchestrate("TCS", "Quarterly results", force_refresh=True)
"""

import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_plane.config import Jurisdiction
from control_plane.manager import ControlPlaneManager, DataChecklist, ControlPlaneResult
from inference_plane.reader import InferenceReader, RetrievalResult


@dataclass
class OrchestrateResult:
    """Complete result from the orchestration pipeline."""
    ticker: str
    jurisdiction: Optional[str]
    query: str

    # Control Plane results
    folder_existed: bool
    components_checked: dict[str, bool]
    components_updated: list[str]
    components_indexed: list[str]
    control_plane_errors: list[str]

    # Inference Plane results
    retrieval_matches: list[dict]
    retrieval_context: str

    @property
    def success(self) -> bool:
        """True if we have some results to return."""
        return len(self.retrieval_matches) > 0 or len(self.control_plane_errors) == 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "jurisdiction": self.jurisdiction,
            "query": self.query,
            "control_plane": {
                "folder_existed": self.folder_existed,
                "components_checked": self.components_checked,
                "components_updated": self.components_updated,
                "components_indexed": self.components_indexed,
                "errors": self.control_plane_errors
            },
            "retrieval": {
                "num_matches": len(self.retrieval_matches),
                "matches": self.retrieval_matches,
                "context": self.retrieval_context
            }
        }


def orchestrate(
    ticker: str,
    query: str,
    checklist: Optional[DataChecklist] = None,
    cik: Optional[str] = None,
    scrip_code: Optional[str] = None,
    force_refresh: bool = False,
    top_k: int = 5
) -> OrchestrateResult:
    """
    Full pipeline: Control Plane → Inference Plane.

    1. Control Plane ensures data is fresh and indexed
    2. Inference Plane queries Pinecone and returns results

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TCS")
        query: User's query for retrieval
        checklist: Which data components to ensure (default: all)
        cik: CIK for US companies (if not in registry)
        scrip_code: Scrip code for Indian companies (if not in registry)
        force_refresh: Force refetch regardless of freshness
        top_k: Number of retrieval results to return

    Returns:
        OrchestrateResult with control plane and retrieval results
    """
    ticker = ticker.upper()

    print(f"\n{'='*60}")
    print(f"ORCHESTRATING: {ticker}")
    print(f"Query: {query[:50]}..." if len(query) > 50 else f"Query: {query}")
    print(f"{'='*60}\n")

    # ===== CONTROL PLANE =====
    print("[1/2] Control Plane: Ensuring data is ready...")
    manager = ControlPlaneManager()
    control_result: ControlPlaneResult = manager.ensure_data_ready(
        ticker=ticker,
        checklist=checklist,
        cik=cik,
        scrip_code=scrip_code,
        force_refresh=force_refresh
    )

    if control_result.errors:
        print(f"  Warnings/Errors: {control_result.errors}")

    jurisdiction_str = control_result.jurisdiction.value if control_result.jurisdiction else None

    # ===== INFERENCE PLANE =====
    print("\n[2/2] Inference Plane: Querying Pinecone...")
    retrieval_matches = []
    retrieval_context = ""

    try:
        reader = InferenceReader()
        retrieval_result: RetrievalResult = reader.retrieve(
            query=query,
            ticker=ticker,
            top_k=top_k
        )

        retrieval_matches = [
            {
                "id": m.id,
                "score": m.score,
                "text": m.text,
                "metadata": m.metadata
            }
            for m in retrieval_result.matches
        ]
        retrieval_context = retrieval_result.get_context()

        print(f"  Found {len(retrieval_matches)} matches")
        for i, m in enumerate(retrieval_result.matches[:3], 1):
            print(f"    {i}. {m}")

    except Exception as e:
        control_result.errors.append(f"Retrieval error: {str(e)}")
        print(f"  Retrieval error: {e}")

    print(f"\n{'='*60}")
    print(f"ORCHESTRATION COMPLETE")
    print(f"{'='*60}\n")

    return OrchestrateResult(
        ticker=ticker,
        jurisdiction=jurisdiction_str,
        query=query,
        folder_existed=control_result.folder_existed,
        components_checked=control_result.components_checked,
        components_updated=control_result.components_updated,
        components_indexed=control_result.components_indexed,
        control_plane_errors=control_result.errors,
        retrieval_matches=retrieval_matches,
        retrieval_context=retrieval_context
    )


def control_only(
    ticker: str,
    checklist: Optional[DataChecklist] = None,
    cik: Optional[str] = None,
    scrip_code: Optional[str] = None,
    force_refresh: bool = False
) -> ControlPlaneResult:
    """
    Run only the Control Plane (no retrieval).

    Use this when you just want to ensure data is fresh/indexed
    without performing a query.

    Args:
        ticker: Stock ticker symbol
        checklist: Which data components to manage
        cik: CIK for US companies
        scrip_code: Scrip code for Indian companies
        force_refresh: Force refetch regardless of freshness

    Returns:
        ControlPlaneResult with operation details
    """
    manager = ControlPlaneManager()
    return manager.ensure_data_ready(
        ticker=ticker,
        checklist=checklist,
        cik=cik,
        scrip_code=scrip_code,
        force_refresh=force_refresh
    )


def retrieve_only(
    ticker: str,
    query: str,
    top_k: int = 5
) -> RetrievalResult:
    """
    Run only the Inference Plane (no data management).

    Use this when you know data is already fresh/indexed
    and just want to query.

    Args:
        ticker: Stock ticker symbol
        query: User's query
        top_k: Number of results

    Returns:
        RetrievalResult with matching documents
    """
    reader = InferenceReader()
    return reader.retrieve(query=query, ticker=ticker, top_k=top_k)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Financial Analysis Pipeline Orchestrator"
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, TCS)")
    parser.add_argument("query", help="Query for retrieval")
    parser.add_argument("--cik", help="SEC CIK for US companies")
    parser.add_argument("--scrip", help="BSE scrip code for Indian companies")
    parser.add_argument("--force", action="store_true", help="Force refresh data")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    result = orchestrate(
        ticker=args.ticker,
        query=args.query,
        cik=args.cik,
        scrip_code=args.scrip,
        force_refresh=args.force,
        top_k=args.top_k
    )

    print("\n--- RESULT ---")
    print(f"Ticker: {result.ticker} ({result.jurisdiction})")
    print(f"Components Updated: {result.components_updated}")
    print(f"Matches Found: {len(result.retrieval_matches)}")

    if result.control_plane_errors:
        print(f"Errors: {result.control_plane_errors}")

    print("\n--- CONTEXT FOR LLM ---")
    print(result.retrieval_context[:2000] if result.retrieval_context else "No context available")
