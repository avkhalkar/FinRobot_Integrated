import sys
from typing import List, Set
from pathlib import Path

# --- DYNAMIC PATH RESOLUTION ---
# Explicitly adding project root to sys.path to fix ModuleNotFoundError
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# -------------------------------

# Internal Imports
from agent.schemas import Chunk
from utils.llm_client import generate_secondary, get_embedding
from utils.similarity import cosine_similarity
from config.token_budgets import TOKEN_BUDGETS
from utils.logger import get_logger

logger = get_logger("CONTEXT_COMPRESSOR")

# Constants
MAX_RETRIES = 3
SIMILARITY_THRESHOLD = 0.9  # Chunks >90% similar are considered duplicates

COMPRESSION_PROMPT = (
    "You are a Precision Editor for financial data."
    "Your task is to COMPRESS the provided text by removing conversational filler,"
    "redundant adjectives, and formatting noise."
    "CRITICAL PRESERVATION RULES:"
    "1. PRESERVE EXACTLY all numbers, percentages, dates, currency values, and proper nouns."
    "2. PRESERVE all conditionality (if/then, unless, except)."
    "3. PRESERVE table structures if present."
    "4. DO NOT round numbers or interpret legal clauses."
    "5. Target length: roughly {max_tokens} tokens (but prioritize accuracy over length)."
)

def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Uses Vector Similarity to remove redundant context chunks.
    If Chunk A and Chunk B are >90% similar, we drop the redundant one.
    """
    if not chunks:
        return []

    logger.info(f"Deduplicating {len(chunks)} chunks...")
    unique_chunks = []
    
    # 1. Generate Embeddings for all chunks
    # Note: In a real prod env, these might already be in chunk.metadata, 
    # but we fetch them here to be safe.
    embeddings = []
    for chunk in chunks:
        # We use the text content for similarity comparison
        emb = get_embedding(chunk.text)
        embeddings.append(emb)

    indices_to_drop: Set[int] = set()

    # 2. Compare O(N^2) - okay for small N (top-k=5 to 10)
    for i in range(len(chunks)):
        if i in indices_to_drop:
            continue
            
        for j in range(i + 1, len(chunks)):
            if j in indices_to_drop:
                continue
            
            sim = cosine_similarity(embeddings[i], embeddings[j])
            
            if sim > SIMILARITY_THRESHOLD:
                logger.debug(f"Duplicate found: Chunk {chunks[i].id} vs {chunks[j].id} (Sim: {sim:.2f})")
                
                # Heuristic: Keep the longer text (more info)
                if len(chunks[i].text) >= len(chunks[j].text):
                    indices_to_drop.add(j)
                else:
                    indices_to_drop.add(i)
                    break # Stop comparing i, it is dropped

    # 3. Reconstruct List
    for index, chunk in enumerate(chunks):
        if index not in indices_to_drop:
            unique_chunks.append(chunk)

    logger.info(f"Deduplication complete. Reduced {len(chunks)} -> {len(unique_chunks)} chunks.")
    return unique_chunks

def compress_context(chunks: List[Chunk]) -> List[Chunk]:
    """
    Pipeline:
    1. Deduplicate chunks using cosine similarity.
    2. Perform abstractive summarization on remaining long chunks.
    """
    # Step 1: Deduplicate
    clean_chunks = deduplicate_chunks(chunks)
    
    compressed_chunks = []
    max_tokens = TOKEN_BUDGETS.get("COMPRESSION_TARGET_MAX", 500)
    
    # Step 2: Compress if necessary
    for chunk in clean_chunks:
        # Simple heuristic: only compress documents that are likely "long docs"
        # 1 token ~= 4 chars. If text > 1.5x budget, compress.
        if len(chunk.text.split()) < (max_tokens * 1.5):
            compressed_chunks.append(chunk)
            continue

        logger.info(f"Compressing Chunk {chunk.id} (Length: {len(chunk.text)} chars)...")
        
        summary = None
        
        # Retry Logic
        for attempt in range(MAX_RETRIES):
            prompt_content = f"Original Text:\n{chunk.text}"
            
            summary = generate_secondary(
                system_prompt=COMPRESSION_PROMPT.format(max_tokens=max_tokens),
                user_prompt=prompt_content,
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            # Successful generation check
            if not summary.startswith("ERROR") and summary.strip():
                logger.debug(f"Chunk {chunk.id} compressed successfully on attempt {attempt + 1}.")
                break
            
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: Compression failed. Retrying...")
            
        # Final Fallback
        if summary is None or summary.startswith("ERROR") or not summary.strip():
            logger.error(f"Compression failed for chunk {chunk.id}. Retaining original.")
            compressed_chunks.append(chunk)
        else:
            # Create a new chunk object for the compressed version
            compressed_chunk = Chunk(
                id=chunk.id, 
                text=summary, 
                metadata={**chunk.metadata, "compression_status": "compressed"}
            )
            compressed_chunks.append(compressed_chunk)
            logger.info(f"Chunk {chunk.id} compressed. New length: {len(summary)} chars.")
            
    return compressed_chunks