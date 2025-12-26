# utils/logger.py
import logging
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from config.settings import settings

def setup_logging():
    """
    Initializes the root logger with settings from the config.
    This should be called EXACTLY ONCE at the start of your application 
    (e.g., in main_agent.py).
    """
    # 1. robustly fetch the log level string (e.g., "DEBUG")
    log_level_str = settings.LOG_LEVEL.upper()
    
    # 2. Convert string to logging constant (default to INFO if invalid)
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    # 3. Configure the logging system
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        # force=True allows re-configuration if a library set it up earlier
        force=True 
    )

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for a given module name.
    Prefixes with AGENT. for consistent naming in logs.
    
    Example: get_logger("planner") -> "AGENT.planner"
    """
    return logging.getLogger(f"AGENT.{name}")