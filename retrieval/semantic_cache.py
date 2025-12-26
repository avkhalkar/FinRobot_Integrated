# retrieval/semantic_cache.py
from typing import Optional, Dict, Any
import time
import sys
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from retrieval.pinecone_client import index
from utils.llm_client import get_embedding
from utils.logger import get_logger
from agent.schemas import UserProfileSchema

logger = get_logger("SEMANTIC_CACHE")

CACHE_NAMESPACE = "semantic_cache"
CACHE_THRESHOLD = 0.98  # High threshold for exact semantic matches

def clear_cache() -> bool:
    """Wipes the cache (use when docs are updated)."""
    if not index: return False
    try:
        index.delete(delete_all=True, namespace=CACHE_NAMESPACE)
        logger.info("Semantic cache cleared.")
        return True
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return False

def store_cache(query: str, response: str, profile: Optional[UserProfileSchema] = None) -> None:
    """
    Stores response with Profile Metadata (Style, Depth, Risk) to ensure context-aware retrieval.
    The cache entry is 'stamped' with the personality settings active at the time of generation.
    """
    if not response or not query or not index: 
        return

    try:
        vector = get_embedding(query)
        if not vector: return

        # Default metadata
        metadata = {
            "query": query,
            "cached_response": response,
            "timestamp": time.time()
        }

        # Add Profile Context if available
        if profile:
            metadata.update({
                "style": profile.style_preference,       # e.g., 'formal', 'casual'
                "depth": profile.explanation_depth,      # e.g., 'detailed', 'simple'
                "risk": profile.risk_tolerance           # e.g., 'low', 'high'
            })

        # ID is hash of query + profile config to allow unique entries for different personalities
        # (Simple hash of query is usually enough if we filter, but unique IDs help debugging)
        import hashlib
        id_str = f"{query}_{profile.style_preference if profile else 'default'}_{profile.explanation_depth if profile else 'default'}"
        unique_id = hashlib.md5(id_str.encode()).hexdigest()

        index.upsert(
            vectors=[(unique_id, vector, metadata)],
            namespace=CACHE_NAMESPACE
        )
        logger.debug(f"Stored in cache: {unique_id[:8]}...")

    except Exception as e:
        logger.error(f"Cache store failed: {e}")

def retrieve_cache(query: str, profile: Optional[UserProfileSchema] = None) -> Optional[str]:
    """
    Retrieves a cached answer ONLY if it matches the current User Profile settings.
    This prevents returning a 'simple' answer when the user has switched to 'technical'.
    """
    if not index or not query: return None

    try:
        query_vector = get_embedding(query)
        if not query_vector: return None

        # Build Strict Filter
        # We only accept cache hits that match the user's CURRENT needs.
        filter_dict: Dict[str, Any] = {}
        
        if profile:
            filter_dict = {
                "style": profile.style_preference,
                "depth": profile.explanation_depth,
                "risk": profile.risk_tolerance
            }

        # Query Pinecone with filter
        results = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            namespace=CACHE_NAMESPACE,
            filter=filter_dict if filter_dict else None  # <--- The Smart Filter
        )

        matches = results.get('matches', [])
        if not matches:
            return None

        match = matches[0]
        if match.score >= CACHE_THRESHOLD:
            logger.info(f"CACHE HIT ({match.score:.4f}) | Filter Used: {filter_dict}")
            return match.metadata.get('cached_response')
        
        return None

    except Exception as e:
        logger.error(f"Cache retrieval failed: {e}")
        return None