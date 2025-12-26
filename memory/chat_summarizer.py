# memory/chat_summarizer.py
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from google.genai import types
from agent.schemas import ChatSummary, UserProfileSchema
from utils.llm_client import generate_secondary
from utils.json_utils import safe_json_load
from utils.logger import get_logger

logger = get_logger("CHAT_SUMMARIZER")

class ChatSummarizer:
    """
    The Cortex: Analyzes interactions to evolve the User Profile in real-time
    and compresses history for long-term memory.
    """
    MAX_CONTEXT_CHARS = 12000 
    MAX_PREFERENCES = 15 # Hard limit for list size

    def _get_truncated_text(self, history: List[str]) -> str:
        full_text = "\n".join(history)
        if len(full_text) > self.MAX_CONTEXT_CHARS:
            return "..." + full_text[-self.MAX_CONTEXT_CHARS:]
        return full_text

    def analyze_interaction_delta(self, current_profile: UserProfileSchema, last_user_msg: str, last_agent_msg: str) -> Dict[str, Any]:
        """
        Scans a SINGLE interaction to intelligently evolve the profile.
        Handles merging, deduplication, and compaction of lists/summaries server-side (LLM).
        """
        
        # Serialize current state explicitly for the prompt
        curr_prefs = ", ".join(current_profile.preferences) if current_profile.preferences else "None"
        curr_bugs = current_profile.prior_misunderstandings_summary if current_profile.prior_misunderstandings_summary else "None"
        
        system_prompt = (
            "You are a Real-Time User Profile Architect.\n"
            "Your goal is to EVOLVE the user profile based on the LATEST INTERACTION.\n\n"
            
            "--- RULES FOR SCALAR FIELDS (Update only if explicit change detected) ---\n"
            "1. Risk Tolerance: 'low' (safety/guarantees), 'medium', 'high' (aggressive/moonshots).\n"
            "2. Explanation Depth: 'simple' (ELI5), 'detailed', 'technical' (code/math).\n"
            "3. Style Preference: 'formal', 'casual', 'concise'.\n\n"
            
            "--- RULES FOR COMPOSITE FIELDS (Smart Evolution) ---\n"
            f"4. Preferences (Max {self.MAX_PREFERENCES} items):\n"
            f"   - CURRENT LIST: [{curr_prefs}]\n"
            "   - If new interests appear: MERGE them into the current list.\n"
            "   - DEDUPLICATE: Consolidate similar items (e.g., 'Python' + 'Python coding' -> 'Python').\n"
            "   - PRIORITIZE: Keep the list concise. If > 15 items, drop the least relevant/oldest.\n"
            "   - OUTPUT: Return the FULL REPLACEMENT list.\n\n"
            
            "5. Prior Misunderstandings Summary:\n"
            f"   - CURRENT SUMMARY: {curr_bugs}\n"
            "   - If the user corrects the agent: REWRITE the summary to concisely explain what NOT to do.\n"
            "   - MERGE the new correction with the old summary. Do not just append. Synthesize.\n"
            "   - OUTPUT: Return the FULL REPLACEMENT string.\n\n"

            "OUTPUT FORMAT:\n"
            "Return JSON. Include ONLY fields that have changed/evolved. If no change, return empty JSON."
        )

        interaction_text = f"User: {last_user_msg}\nAgent: {last_agent_msg}"

        schema = {
            "type": types.Type.OBJECT,
            "properties": {
                "risk_tolerance": {"type": types.Type.STRING, "enum": ["low", "medium", "high"]},
                "explanation_depth": {"type": types.Type.STRING, "enum": ["simple", "detailed", "technical"]},
                "style_preference": {"type": types.Type.STRING, "enum": ["formal", "casual", "concise"]},
                "prior_misunderstandings_summary": {"type": types.Type.STRING},
                "preferences": {
                    "type": types.Type.ARRAY,
                    "items": {"type": types.Type.STRING}
                }
            },
            "required": [] 
        }

        try:
            raw_response = generate_secondary(
                system_prompt=system_prompt,
                user_prompt=f"LATEST INTERACTION:\n{interaction_text}",
                response_schema=schema,
                temperature=0.0 # Strict logic
            )

            updates = raw_response if isinstance(raw_response, dict) else safe_json_load(raw_response)
            
            if not updates: 
                return {}

            # Final Safety Guard: Python-side truncation just in case LLM slipped
            if "preferences" in updates and isinstance(updates["preferences"], list):
                if len(updates["preferences"]) > self.MAX_PREFERENCES:
                    updates["preferences"] = updates["preferences"][:self.MAX_PREFERENCES]

            return updates

        except Exception as e:
            logger.error(f"Profile evolution analysis failed: {e}")
            return {}

    def summarize(self, conversation_history: List[str]) -> ChatSummary:
        """Standard episodic compression."""
        if not conversation_history:
            return ChatSummary(summary="", key_facts=[])

        text_block = self._get_truncated_text(conversation_history)
        
        system_prompt = (
            "Summarize the conversation. "
            "Extract key facts (dates, numbers, entities) and a 2-sentence summary."
        )
        
        schema = {
            "type": types.Type.OBJECT,
            "properties": {
                "summary": {"type": types.Type.STRING},
                "key_facts": {"type": types.Type.ARRAY, "items": {"type": types.Type.STRING}}
            },
            "required": ["summary", "key_facts"]
        }

        try:
            raw_response = generate_secondary(
                system_prompt=system_prompt,
                user_prompt=f"CONVERSATION:\n{text_block}",
                response_schema=schema,
                temperature=0.0 
            )
            data = raw_response if isinstance(raw_response, dict) else safe_json_load(raw_response)
            return ChatSummary.model_validate(data) if data else ChatSummary(summary="", key_facts=[])
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ChatSummary(summary="Error.", key_facts=[])

    def deduplicate_facts(self, existing_facts: List[str], new_candidates: List[str]) -> List[str]:
        """
        Prevents long-term memory bloat by maintaining a strict, high-value list of facts.
        """
        if not new_candidates and not existing_facts:
            return []

        # STRICT LIMIT: 25 Facts
        system_prompt = (
            "You are a Memory Optimizer. Consolidate facts about the user.\n"
            "Input: EXISTING_FACTS and NEW_CANDIDATES.\n"
            "Task: Create a SINGLE, merged list of facts.\n"
            "--- GUIDELINES ---\n"
            "1. **STRICT LIMIT**: The output list must contain NO MORE THAN 25 items.\n"
            "2. **STYLE**: Concise, detailed, no blabber. Absolute information density.\n"
            "3. **MERGE**: If existing says 'User is 20' and new says 'User is 21', keep only 'User is 21'.\n"
            "4. **PRIORITIZE**: If > 25 facts exist, discard the oldest or least relevant (e.g., trivial likes) to fit the limit.\n"
            "5. **OUTPUT**: Return the FINAL complete list of facts to be stored.\n"
        )

        user_prompt = f"EXISTING_FACTS: {existing_facts}\nNEW_CANDIDATES: {new_candidates}"
        
        schema = {
            "type": types.Type.OBJECT,
            "properties": {
                "final_facts_list": {"type": types.Type.ARRAY, "items": {"type": types.Type.STRING}}
            },
            "required": ["final_facts_list"]
        }

        try:
            response = generate_secondary(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=schema,
                temperature=0.0
            )
            data = response if isinstance(response, dict) else safe_json_load(response)
            
            # Defensive check
            final_list = data.get("final_facts_list", [])
            if len(final_list) > 25:
                logger.warning("LLM returned > 25 facts. Truncating.")
                return final_list[:25]
            return final_list
            
        except Exception as e:
            logger.error(f"Fact deduplication failed: {e}")
            # Fallback: Combine and hard truncate to safety limit
            combined = list(set(existing_facts + new_candidates))
            return combined[:25]