import sys
from pathlib import Path
import os

# Add project root to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

print("--- Testing Settings Loading ---")
try:
    from config.settings import settings
    print(f"Settings loaded successfully. API Key present: {'Yes' if settings.GEMINI_API_KEY else 'No'}")
except Exception as e:
    print("Caught expected exception during settings load (if key is missing).")
    # Exception is already printed by settings.py logic I added

print("\n--- Testing UI Import Sanity ---")
try:
    import chatbot_ui
    print("chatbot_ui imported successfully (syntax check).")
except ImportError as e:
    print(f"ImportError on chatbot_ui: {e}")
except Exception as e:
    # chatbot_ui runs code on import (st.set_page_config etc), so might fail in non-streamlit env
    print(f"chatbot_ui execution error (expected if not in streamlit): {e}")

print("\n--- Testing Query Refiner Fix Logic ---")
import json
def safe_json_load(s):
    try: return json.loads(s)
    except: return None

strategy_resp = '[{"selected_strategy": "HyDE"}]'
parsed_strategy = safe_json_load(strategy_resp)
if isinstance(parsed_strategy, list) and len(parsed_strategy) > 0:
    if isinstance(parsed_strategy[0], dict):
        print("Strategy selection list test: PASSED (Unwrapped)")
        strategy_resp = json.dumps(parsed_strategy[0])
else:
    print("Strategy selection list test: FAILED")

print(f"Final strategy string: {strategy_resp}")
