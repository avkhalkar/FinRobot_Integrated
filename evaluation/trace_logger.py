# evaluation/trace_logger.py
import json
import uuid
import datetime
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import GlobalState
from utils.logger import get_logger

logger = get_logger("TRACE_LOGGER")

class TraceLogger:
    """
    Standardized Audit Recorder (Black Box).
    Ensures every cognitive cycle is persisted as a recoverable JSON trace.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = project_root / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TraceLogger initialized at {self.log_dir}")

    def log_cycle(self, state: GlobalState):
        """
        Serializes the full GlobalState. 
        Uses UUID and ISO format for unique, sortable filenames.
        """
        try:
            # ISO format for alphabetical sorting in file explorers
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            run_id = str(uuid.uuid4())[:8]
            filename = f"trace_{timestamp}_{run_id}.json"
            filepath = self.log_dir / filename

            # Use model_dump_json for Pydantic-native serialization (handles datetimes, types)
            json_payload = state.model_dump_json(indent=2)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_payload)
                
            logger.info(f"Audit trace persisted: {filename} (Iterations: {len(state.iteration_log)})")
        except Exception as e:
            logger.error(f"Critical Logging Failure: {e}")