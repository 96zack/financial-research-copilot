from financial_research_copilot.models.chunk import (
    ChunkMetadata,
    DocumentChunk,
)


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            cik="0000320193",
            company_name="Apple Inc.",
            form="10-K",
            filing_date="2026-10-30",
            source_url="https://www.sec.gov/",
            chunk_index=0,
        ),
    )
