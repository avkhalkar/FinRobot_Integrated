# evaluation/ragas_runner.py
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from google.genai import types

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.llm_client import generate_secondary, get_embedding
from utils.json_utils import safe_json_load
from utils.similarity import cosine_similarity
from utils.logger import get_logger

logger = get_logger("RAGAS_RUNNER")

class MetricResult(BaseModel):
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class RagasRunner:
    def __init__(self):
        self.schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "metric_name": types.Schema(type=types.Type.STRING),
                "score": types.Schema(type=types.Type.NUMBER),
                "reasoning": types.Schema(type=types.Type.STRING),
            },
            required=["metric_name", "score", "reasoning"]
        )

    def _call_metric(self, sys_p: str, usr_p: str) -> MetricResult:
        res = generate_secondary(sys_p, usr_p, response_schema=self.schema)
        data = safe_json_load(res)
        if data:
            return MetricResult.model_validate(data)
        return MetricResult(metric_name="error", score=0.0, reasoning="Parsing failure.")

    def evaluate_faithfulness(self, context: List[str], answer: str) -> MetricResult:
        prompt = f"Is the answer grounded in context? Context: {' '.join(context)} | Answer: {answer}"
        return self._call_metric("Evaluation: Faithfulness (0-1)", prompt)

    # UPDATED: Added chat_history for context-aware relevance
    def evaluate_answer_relevance(self, query: str, answer: str, chat_history: Optional[str] = None) -> MetricResult:
        context_str = f"Chat History: {chat_history}\n" if chat_history else ""
        prompt = (
            f"Determine if the answer is relevant to the query given the history.\n"
            f"{context_str}"
            f"Query: {query} | Answer: {answer}"
        )
        return self._call_metric("Evaluation: Relevance (0-1)", prompt)

    def evaluate_semantic_similarity(self, query: str, answer: str) -> MetricResult:
        try:
            v1, v2 = get_embedding(query), get_embedding(answer)
            score = cosine_similarity(v1, v2) if v1 and v2 else 0.0
            return MetricResult(metric_name="semantic_similarity", score=score, reasoning="Vector cosine distance.")
        except Exception as e:
            logger.error(f"Sim-Eval Error: {e}")
            return MetricResult(metric_name="semantic_similarity", score=0.0, reasoning="Vector error.")

    # UPDATED: Accept chat_history in the orchestration loop
    def run_full_evaluation(self, query: str, context: List[str], answer: str, chat_history: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Initiating RAG evaluation cycle...")
        
        f = self.evaluate_faithfulness(context, answer)
        r = self.evaluate_answer_relevance(query, answer, chat_history)
        s = self.evaluate_semantic_similarity(query, answer)
        
        overall = (f.score * 0.5) + (r.score * 0.3) + (s.score * 0.2)
        
        return {
            "metrics": {"faithfulness": f.dict(), "relevance": r.dict(), "semantic": s.dict()},
            "overall_score": round(overall, 4)
        }