from financial_research_copilot.ingestion.document_pipeline import (
    ingest_and_chunk_filing,
)

result = ingest_and_chunk_filing(
    cik="0000320193",
    form_type="10-K",
    user_agent="Your Name your-email@example.com",
)

print("Company:", result["filing"]["company_name"])
print("Filing date:", result["filing"]["filing_date"])
print("Chunk count:", len(result["chunks"]))
print("Metrics:", result["metrics"])

first_chunk = result["chunks"][0]

print("\nFirst chunk ID:", first_chunk.chunk_id)
print("First chunk metadata:", first_chunk.metadata.model_dump())
print("First chunk preview:", repr(first_chunk.text[:200]))

print(
    "\nFooter check:",
    any("Apple Inc. | 2026 Form 10-K |" in chunk.text for chunk in result["chunks"]),
)
