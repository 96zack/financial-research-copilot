import pytest

from financial_research_copilot.models.chunk import (
    ChunkMetadata,
    DocumentChunk,
)
from financial_research_copilot.processing.deduplicator import (
    deduplicate_chunks,
)


@pytest.fixture
def make_chunk():
    def _make_chunk(chunk_id: str, text: str) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                company_name="Apple Inc.",
                cik="0000320193",
                form="10-K",
                filing_date="2026-09-26",
                source_url="https://example.com/apple-10k",
                chunk_index=int(chunk_id),
            ),
        )

    return _make_chunk


def test_removes_duplicate_text(make_chunk):
    chunk_a1 = make_chunk("001", "Apple relies on suppliers.")
    chunk_b = make_chunk("002", "Apple faces AI risks.")
    chunk_a2 = make_chunk("003", "Apple relies on suppliers.")

    result = deduplicate_chunks([chunk_a1, chunk_b, chunk_a2])

    assert len(result) == 2
    assert result[0].chunk_id == "001"
    assert result[1].chunk_id == "002"
    assert result[0] is chunk_a1


def test_preserves_unique_chunks(make_chunk):
    chunks = [
        make_chunk("001", "Apple relies on suppliers."),
        make_chunk("002", "Apple faces AI risks."),
        make_chunk("003", "Apple reports gross margins."),
    ]

    result = deduplicate_chunks(chunks)

    assert result == chunks


def test_empty_input():
    assert deduplicate_chunks([]) == []
