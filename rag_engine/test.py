from src.orchestrate import orchestrate
from src.control_plane.manager import DataChecklist

# US Company - checks freshness, fetches if stale, indexes, retrieves
# result = orchestrate("AAPL", "What are Apple's risk factors?")

# Indian Company
# result = orchestrate("TCS", "Quarterly results", scrip_code="532540")

# # Force refresh data
# result = orchestrate("MSFT", "Revenue trends", force_refresh=True)


# # Access results
# print(result.retrieval_context)      # Context for LLM
# print(result.components_updated)     # What was refetched
# print(result.retrieval_matches)      # Raw matches with scores

# For unstructured data of Indian companies

result = orchestrate(
    ticker="TCS", 
    query="What are the key risks mentioned in the latest report?", 
    scrip_code="532540",
    checklist=DataChecklist(structured=[], unstructured=True) 
)
# Access results
print(f"Context Length: {len(result.retrieval_context)}")
print(result.retrieval_context)