# retrieval/pinecone_client.py
from typing import List, Dict, Any, Optional, Set
from pinecone import Pinecone
import sys
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from utils.llm_client import get_embedding
from utils.logger import get_logger
from agent.schemas import Chunk

# --- Smart Retrieval Imports ---
try:
    from retrieval.query_refiner import refine_query
    from retrieval.context_compressor import compress_context
    SMART_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Smart components import failed: {e}")
    SMART_COMPONENTS_AVAILABLE = False

logger = get_logger("PINECONE_CLIENT")

# --- RAG Integration ---
try:
    from rag_engine.src.orchestrate import orchestrate
    from rag_engine.src.control_plane.manager import DataChecklist
    RAG_AVAILABLE = True
except ImportError as e:
    logger.critical(f"RAG Engine import failed: {e}")
    RAG_AVAILABLE = False

# Remove direct Pinecone Init as rag_engine handles it
pc: Optional[Pinecone] = None
index: Any = None

def extract_ticker(query: str) -> str:
    """
    Simple heuristic to find a ticker in the query.
    Defaults to AAPL if none found (Temporary for integration).
    """
    # Common tech tickers for demo
    # Ordered by length to match specific longer tickers first
    tickers = ["GOOGL", "MSFT", "AAPL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "TCS", "INFY"]
    query_upper = query.upper()
    
    # Check for Apple specifically as it's common
    if "APPLE" in query_upper:
        return "AAPL"
        
    for t in tickers:
        if t in query_upper:
            return t
            
    # Fallback to AAPL if no known ticker found
    return "AAPL"

def retrieve(
    query: str, 
    filters: Optional[Dict[str, Any]] = None, 
    namespace: str = "", 
    top_k: int = settings.RAG_TOP_K
) -> List[Chunk]:
    """
    Proxies the request to the Financial Analysis RAG Engine.
    """
    if not RAG_AVAILABLE:
        logger.error("RAG Engine not available.")
        return []

    if not query:
        logger.warning("Empty query provided to retrieve().")
        return []

    ticker = extract_ticker(query)
    logger.info(f"Delegating retrieval to RAG Engine for Ticker: {ticker}")

    try:
        # Orchestrate handles Freshness + Retrieval
        # We perform a 'Retrieve Only' logical flow but via the orchestrator to ensure data is there.
        # Check if we should enforce unstructured data (e.g. if query mentions 'filing' or 'report')
        checklist = DataChecklist(structured=["price"], unstructured=True)
        
        result = orchestrate(
            ticker=ticker,
            query=query,
            checklist=checklist,
            top_k=top_k
        )
        
        chunks = []
        for match in result.retrieval_matches:
            chunks.append(
                Chunk(
                    id=match.get('id', 'unknown'),
                    text=match.get('text', ''),
                    score=match.get('score', 0.0),
                    metadata=match.get('metadata', {})
                )
            )
        
        logger.info(f"RAG Engine returned {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logger.error(f"RAG Engine failure: {e}")
        return []

def smart_retrieve(query: str, chat_history: str = "None") -> List[Chunk]:
    """
    Orchestrates Retrieval.
    1. Always includes the RAW query to ensure basic keyword/semantic matching works.
    2. Refines query using chat history.
    3. Fetches chunks.
    4. Compresses chunks based on the USER QUERY before returning.
    """
    # 1. Fallback if smart components are missing
    if not SMART_COMPONENTS_AVAILABLE:
        return retrieve(query)

    try:
        # 2. Prepare Query List
        # We start with the raw query to guarantee at least basic retrieval performance
        queries = [query]
        
        # Attempt refinement
        try:
            refined = refine_query(query, chat_history)
            if refined and isinstance(refined, list):
                queries.extend(refined)
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Proceeding with raw query only.")

        # 3. Execution (Multi-Query)
        all_chunks: List[Chunk] = []
        seen_ids: Set[str] = set()
        
        logger.info(f"Executing smart retrieval with {len(queries)} query variations.")
        
        for q in queries:
            # We use a slightly lower top_k for variations to keep context focused
            results = retrieve(q, top_k=3) 
            for chunk in results:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)
        
        logger.info(f"Smart Retrieve found {len(all_chunks)} unique chunks before compression.")
        
        # 4. Context Compression & Deduplication (Query Aware)
        # We pass the original user query to ensure compression preserves relevant info
        if all_chunks:
            return compress_context(all_chunks, query)
            
        return all_chunks

    except Exception as e:
        logger.error(f"Smart retrieval critical failure: {e}. Fallback to raw.")
        return retrieve(query)