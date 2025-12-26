# retrieval/query_refiner.py
import sys
from typing import List, Optional
from pathlib import Path
from google.genai import types 

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.llm_client import generate_secondary
from utils.json_utils import safe_json_load
from utils.logger import get_logger

logger = get_logger("QUERY_REFINER")

MAX_RETRIES = 2

# --- Prompts ---

STRATEGY_SELECTOR_PROMPT = (
    "You are a Search Strategist. Analyze the User Query and Chat History to select the "
    "single best retrieval query refinement technique.\n"
    "OPTIONS:\n"
    "1. 'HyDE': Best for vague queries needing a hypothetical answer to ground the vector search.\n"
    "2. 'Decomposition': Best for complex, multi-part questions.\n"
    "3. 'HistoryContext': Best when the query relies heavily on previous chat context (e.g., 'What about him?').\n"
    "4. 'KeywordExpansion': Best for simple, specific technical lookups.\n\n"
    "Output valid JSON with key 'selected_strategy' (string) and 'reasoning' (string)."
)

QUERY_REFINER_PROMPT_TEMPLATE = (
    "You are a specialized Query Refiner using the '{STRATEGY}' technique.\n"
    "Generate 3 diverse search queries to maximize retrieval coverage.\n"
    "Context: {CHAT_HISTORY}\n"
    "User Query: {ORIGINAL_QUERY}\n\n"
    "Output strictly valid JSON with key 'refined_queries' (list of strings)."
)

PLANNER_REFINE_PROMPT = (
    "You are a Query Pre-processor for an AI Planner. Your goal is to rewrite the user's raw query "
    "into a clear, unambiguous objective statement that a Planner agent can easily break down.\n"
    "1. Resolve coreferences using Chat History (e.g., change 'it' to the actual subject).\n"
    "2. Remove conversational noise (e.g., 'Can you please tell me...').\n"
    "3. Make explicit any implied constraints.\n\n"
    "Chat History: {CHAT_HISTORY}\n"
    "Raw Query: {RAW_QUERY}\n\n"
    "Output strictly valid JSON with key 'refined_goal' (string)."
)

def refine_query(original_query: str, chat_history: str = "None") -> List[str]:
    """
    Expands a user query into multiple diverse queries.
    1. Selects the best strategy based on history and query.
    2. Generates variations using that strategy.
    """
    if not original_query:
        return []

    # 1. Select Strategy
    strategy = "HyDE" # Default
    try:
        strategy_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "selected_strategy": types.Schema(type=types.Type.STRING),
                "reasoning": types.Schema(type=types.Type.STRING)
            },
            required=["selected_strategy"]
        )
        
        strategy_resp = generate_secondary(
            system_prompt=STRATEGY_SELECTOR_PROMPT,
            user_prompt=f"Query: {original_query}\nHistory: {chat_history}",
            max_tokens=150,
            response_schema=strategy_schema,
            temperature=0.3
        )
        
        strategy_data = safe_json_load(strategy_resp) if isinstance(strategy_resp, str) else strategy_resp
        if strategy_data and "selected_strategy" in strategy_data:
            strategy = strategy_data["selected_strategy"]
            logger.info(f"Selected Refinement Strategy: {strategy}")

    except Exception as e:
        logger.warning(f"Strategy selection failed, defaulting to KeywordExpansion: {e}")

    # 2. Generate Queries using Strategy
    system_prompt = QUERY_REFINER_PROMPT_TEMPLATE.replace("{STRATEGY}", strategy)
    system_prompt = system_prompt.replace("{CHAT_HISTORY}", chat_history)
    system_prompt = system_prompt.replace("{ORIGINAL_QUERY}", original_query)

    # Strict Schema for Queries
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "refined_queries": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING)
            )
        },
        required=["refined_queries"]
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            raw_response = generate_secondary(
                system_prompt=system_prompt,
                user_prompt="Generate Queries",
                max_tokens=256,
                response_schema=schema,
                temperature=0.7 
            )
            
            refined_data = safe_json_load(raw_response) if isinstance(raw_response, str) else raw_response

            if refined_data and isinstance(refined_data, dict) and "refined_queries" in refined_data:
                generated = refined_data["refined_queries"]
                final_set = list(set([original_query] + generated))
                logger.info(f"Refined '{original_query}' into {len(final_set)} variations.")
                return final_set
            
        except Exception as e:
            logger.warning(f"Refinement attempt {attempt+1} failed: {e}")
    
    logger.error("Query refinement failed. Returning original query.")
    return [original_query]

def refine_query_for_planner(raw_query: str, chat_history: str = "None") -> str:
    """
    Refines a raw user query into a clean, standalone objective for the Planner.
    Resolves ambiguity using chat history.
    """
    if not raw_query:
        return ""

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "refined_goal": types.Schema(type=types.Type.STRING)
        },
        required=["refined_goal"]
    )

    try:
        sys_prompt = PLANNER_REFINE_PROMPT.replace("{CHAT_HISTORY}", chat_history)
        sys_prompt = sys_prompt.replace("{RAW_QUERY}", raw_query)

        raw_response = generate_secondary(
            system_prompt=sys_prompt,
            user_prompt="REFINE GOAL",
            max_tokens=150,
            response_schema=schema,
            temperature=0.2
        )
        
        data = safe_json_load(raw_response) if isinstance(raw_response, str) else raw_response
        
        if data and "refined_goal" in data:
            refined = data["refined_goal"]
            logger.info(f"Planner Query Refined: '{raw_query}' -> '{refined}'")
            return refined
            
    except Exception as e:
        logger.error(f"Planner query refinement failed: {e}")
    
    return raw_query