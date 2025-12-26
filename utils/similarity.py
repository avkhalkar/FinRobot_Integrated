# utils/similarity.py
import numpy as np
from typing import List

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from utils.logger import get_logger

logger = get_logger("SIMILARITY")

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates the cosine similarity between two vectors.
    """
    if not vec_a or not vec_b:
        return 0.0

    try:
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0