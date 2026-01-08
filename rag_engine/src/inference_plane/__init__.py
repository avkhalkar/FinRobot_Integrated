"""
Inference Plane - Reader Layer

Read-only retrieval layer for querying Pinecone.

Invariants:
- NEVER fetches, modifies, re-embeds, or upserts data
- Depends entirely on Control Plane for data correctness/freshness
- Any data inconsistency is a Control Plane failure, not an Inference concern
"""

from .reader import InferenceReader, RetrievalResult

__all__ = [
    "InferenceReader",
    "RetrievalResult",
]
