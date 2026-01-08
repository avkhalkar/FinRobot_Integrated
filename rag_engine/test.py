from src.orchestrate import orchestrate

# US Company - checks freshness, fetches if stale, indexes, retrieves
# result = orchestrate("AAPL", "What are Apple's risk factors?")

# Indian Company
result = orchestrate("TCS", "Quarterly results", scrip_code="532540")

# Force refresh data
# result = orchestrate("MSFT", "Revenue trends", force_refresh=True)



# Access results
print(result.retrieval_context)      # Context for LLM
print(result.components_updated)     # What was refetched
print(result.retrieval_matches)      # Raw matches with scores