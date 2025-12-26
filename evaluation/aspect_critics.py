# evaluation/aspect_critics.py
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from google.genai import types

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.llm_client import generate_secondary
from utils.json_utils import safe_json_load
from utils.logger import get_logger

logger = get_logger("ASPECT_CRITICS")

class CriticScore(BaseModel):
    score: int = Field(..., ge=0, le=10)
    reason: str

class AspectCritics:
    def __init__(self):
        self.response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),
                "reason": types.Schema(type=types.Type.STRING),
            },
            required=["score", "reason"]
        )

    def _evaluate(self, prompt: str) -> CriticScore:
        try:
            json_response = generate_secondary(
                system_prompt="You are a financial auditor. Output valid JSON only.",
                user_prompt=prompt,
                response_schema=self.response_schema
            )
            data = safe_json_load(json_response)
            if data:
                return CriticScore.model_validate(data)
            return CriticScore(score=0, reason="Failed to parse LLM evaluation.")
        except Exception as e:
            logger.error(f"Aspect Critic Logic Failure: {e}")
            return CriticScore(score=0, reason=f"Internal Error: {str(e)}")

    def critique_regulatory_compliance(self, answer: str) -> CriticScore:
        # Regulatory checks are often answer-standalone
        prompt = f"Rate Regulatory Compliance (0-10). Rules: No direct buy/sell advice, neutral tone. Answer: {answer}"
        return self._evaluate(prompt)

    def critique_tone(self, answer: str) -> CriticScore:
        prompt = f"Rate Professional Tone (0-10). 10 is standard banking English. Answer: {answer}"
        return self._evaluate(prompt)

    # UPDATED: Added chat_history to resolve vague queries
    def critique_completeness(self, query: str, answer: str, chat_history: Optional[str] = None) -> CriticScore:
        context_str = f"Chat History: {chat_history}\n" if chat_history else ""
        prompt = (
            f"Given the conversation history and the latest query, rate Completeness (0-10).\n"
            f"{context_str}"
            f"Latest Query: {query}\n"
            f"Answer provided: {answer}\n"
            f"Did the answer address all parts of the user's intent?"
        )
        return self._evaluate(prompt)