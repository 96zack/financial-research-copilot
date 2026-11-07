from financial_research_copilot.processing.chunk_builder import (
    build_document_chunks,
)


def test_build_document_chunks_metadata():
    filing = {
        "company_name": "Apple Inc.",
        "cik": "320193",
        "form": "10-K",
        "filing_date": "2026-10-30",
        "url": "https://example.com/apple-10k",
    }

    chunks = ["First paragraph", "Second paragraph"]

    result = build_document_chunks(filing, chunks)

    assert len(result) == 2

    assert result[0].text == "First paragraph"
    assert result[0].metadata.company_name == "Apple Inc."
    assert result[0].metadata.cik == "320193"
    assert result[0].metadata.chunk_index == 0
    assert result[1].metadata.chunk_index == 1
    assert result[0].metadata.source_url == filing["url"]


def test_build_deterministic_id():
    filing = {
        "company_name": "Apple Inc.",
        "cik": "320193",
        "form": "10-K",
        "filing_date": "2026-10-30",
        "url": "https://example.com/apple-10k",
    }

    chunks = ["First paragraph", "Second paragraph"]

    first = build_document_chunks(filing, chunks)
    second = build_document_chunks(filing, chunks)

    assert first[0].chunk_id == second[0].chunk_id
    assert first[0].chunk_id != first[1].chunk_id


def test_build_empty_chunk():
    filing = {
        "company_name": "Apple Inc.",
        "cik": "320193",
        "form": "10-K",
        "filing_date": "2026-10-30",
        "url": "https://example.com/apple-10k",
    }

    chunks = []

    assert build_document_chunks(filing, chunks) == []
