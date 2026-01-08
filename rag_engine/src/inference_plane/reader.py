"""
Inference Reader - Read-only retrieval from Pinecone.

This layer is completely read-only and NEVER:
- Fetches data from external sources
- Modifies local data
- Re-embeds documents
- Upserts to Pinecone

Query Flow:
1. Convert user query to embedding
2. Query Pinecone using ticker namespace
3. Return top-k relevant chunks
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pinecone import Pinecone

from src.embeddings.embedding_provider import embed_query
from src.control_plane.config import PINECONE_INDEX_NAME


@dataclass
class RetrievalMatch:
    """A single retrieval match from Pinecone."""
    id: str
    score: float
    text: str
    metadata: dict

    def __str__(self) -> str:
        text_preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f"[{self.score:.3f}] {text_preview}"


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    query: str
    ticker: str
    matches: list[RetrievalMatch]
    total_matches: int

    @property
    def has_results(self) -> bool:
        return len(self.matches) > 0

    def get_context(self, max_chars: int = 10000) -> str:
        """
        Get concatenated text from matches for LLM context.

        Args:
            max_chars: Maximum characters to include

        Returns:
            Concatenated text from matches, truncated if needed
        """
        texts = []
        char_count = 0

        for match in self.matches:
            if char_count + len(match.text) > max_chars:
                remaining = max_chars - char_count
                if remaining > 100:  # Only add if meaningful
                    texts.append(match.text[:remaining] + "...")
                break
            texts.append(match.text)
            char_count += len(match.text)

        return "\n\n---\n\n".join(texts)


class InferenceReader:
    """
    Read-only retrieval layer.

    NEVER modifies data - only queries Pinecone.
    """

    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def retrieve(
        self,
        query: str,
        ticker: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> RetrievalResult:
        """
        Query Pinecone for relevant chunks.

        Args:
            query: User's query text
            ticker: Stock ticker (used as Pinecone namespace)
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            RetrievalResult with matching documents
        """
        ticker = ticker.upper()

        # Step 1: Embed the query
        query_vector = embed_query(query)

        # Step 2: Query Pinecone
        query_params = {
            "vector": query_vector,
            "top_k": top_k,
            "namespace": ticker,
            "include_metadata": True
        }

        if filter_dict:
            query_params["filter"] = filter_dict

        response = self.index.query(**query_params)

        # Step 3: Parse results
        matches = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            matches.append(RetrievalMatch(
                id=match.get("id", ""),
                score=match.get("score", 0.0),
                text=metadata.get("text", ""),
                metadata=metadata
            ))

        return RetrievalResult(
            query=query,
            ticker=ticker,
            matches=matches,
            total_matches=len(matches)
        )

    def retrieve_by_category(
        self,
        query: str,
        ticker: str,
        category: str,
        top_k: int = 5
    ) -> RetrievalResult:
        """
        Query Pinecone filtered by data category.

        Args:
            query: User's query text
            ticker: Stock ticker
            category: "narrative" (unstructured) or "narrated_numeric" (structured)
            top_k: Number of results

        Returns:
            RetrievalResult with matching documents
        """
        return self.retrieve(
            query=query,
            ticker=ticker,
            top_k=top_k,
            filter_dict={"data_category": category}
        )

    def check_namespace_exists(self, ticker: str) -> bool:
        """
        Check if a ticker namespace has any vectors.

        Args:
            ticker: Stock ticker

        Returns:
            True if namespace has vectors, False otherwise
        """
        ticker = ticker.upper()
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            return ticker in namespaces and namespaces[ticker].get("vector_count", 0) > 0
        except Exception:
            return False

    def get_namespace_stats(self, ticker: str) -> dict:
        """
        Get statistics for a ticker's namespace.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with vector_count and other stats
        """
        ticker = ticker.upper()
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            return namespaces.get(ticker, {"vector_count": 0})
        except Exception:
            return {"vector_count": 0, "error": "Failed to get stats"}
