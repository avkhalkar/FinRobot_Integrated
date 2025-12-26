# config/token_budgets.py

TOKEN_BUDGETS = {
    # Plan should be a tight JSON array. 400 is plenty for even 10 steps.
    "PLANNER_OUTPUT_MAX": 400,    
    
    # Needs room to synthesize multiple retrieved documents.
    "THINKER_DRAFT_MAX": 1500,    
    
    # Verification reports should be concise 'True/False' with brief citations.
    "VERIFIER_REPORT_MAX": 500,   
    
    # Final response to user. 1200 tokens is roughly 3 pages of text. 
    # Usually, users stop reading after 800.
    "EXPLAINER_FINAL_MAX": 1200,  
    
    # Documents longer than 800 tokens often contain 'fluff' for RAG.
    "COMPRESSION_TARGET_MAX": 800, 
}