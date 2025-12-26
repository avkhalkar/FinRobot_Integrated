# utils/json_utils.py
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import get_logger

logger = get_logger("JSON_UTILS")


def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Extracts the first balanced JSON object or array from text.
    Returns the substring or None if no balanced structure is found.
    """
    stack = []
    start_index = None

    for i, ch in enumerate(text):
        if ch in "{[":
            if start_index is None:
                start_index = i
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                continue
            stack.pop()
            if not stack and start_index is not None:
                return text[start_index : i + 1]

    return None


def safe_json_load(raw_string: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Robustly parses a string into a JSON object or list.

    Strategy:
    1. Direct Parse
    2. Parse from markdown code blocks (```json ... ```)
    3. Balanced brace/bracket extraction from raw text

    Returns None on failure (by design).
    """
    if not raw_string:
        return None

    # --- Strategy 1: Fast Path ---
    try:
        return json.loads(raw_string)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: Markdown Code Blocks ---
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    blocks = re.findall(markdown_pattern, raw_string, re.DOTALL)

    for block in blocks:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # --- Strategy 3: Balanced Extraction ---
    candidate = _extract_balanced_json(raw_string)
    if not candidate:
        logger.warning(
            "No balanced JSON object or array found in text snippet: %s...",
            raw_string[:80],
        )
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        snippet = candidate[:120] + "..." if len(candidate) > 120 else candidate
        logger.warning(
            "JSON parsing failed after extraction. Error: %s | Candidate: %s",
            e,
            snippet,
        )
        return None
    except Exception as e:
        logger.error("Unexpected error in safe_json_load: %s", e)
        return None
