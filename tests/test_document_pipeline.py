from unittest.mock import patch

import pytest

from financial_research_copilot.ingestion.document_pipeline import (
    ingest_and_chunk_filing,
)


def test_ingest_and_chunk_filing():
    fake_filing = {
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "form": "10-K",
        "filing_date": "2026-11-03",
        "url": "https://example.com/apple-10k",
        "text": (
            "Supply Chain Risks\n"
            "Apple Inc. | 2026 Form 10-K | 3\n"
            "Apple depends on third-party suppliers."
        ),
    }

    with patch(
        "financial_research_copilot.ingestion.document_pipeline.ingest_latest_filing",
        return_value=fake_filing,
    ):
        result = ingest_and_chunk_filing(
            cik="0000320193",
            form_type="10-K",
            user_agent="test@example.com",
            chunk_size=100,
            chunk_overlap=20,
        )

    # TODO 1: Assert result has the expected three keys
    assert set(result.keys()) == {"filing", "chunks", "metrics"}
    assert len(result["chunks"]) > 0

    # TODO 2: Assert the footer is absent from every chunk
    assert all(
        "Apple Inc. | 2026 Form 10-K | 3" not in chunk.text
        for chunk in result["chunks"]
    )

    # TODO 3: Assert the original filing text still contains the footer
    assert "Apple Inc. | 2026 Form 10-K | 3" in result["filing"]["text"]

    # TODO 4: Assert every chunk has the correct company metadata
    assert all(
        chunk.metadata.company_name == "Apple Inc." for chunk in result["chunks"]
    )

    # TODO 5: Assert every chunk is at most 100 characters
    assert all(len(chunk.text) <= 100 for chunk in result["chunks"])

    # TODO 6: Assert metrics["chunk_count"] equals len(result["chunks"])
    assert result["metrics"]["chunk_count"] == len(result["chunks"])


def test_ingest_and_chunk_filing_failure():
    with patch(
        "financial_research_copilot.ingestion.document_pipeline.ingest_latest_filing",
        side_effect=RuntimeError("SEC request failed"),
    ):
        # TODO: Verify that ingest_and_chunk_filing()
        # raises RuntimeError with the expected message.
        with pytest.raises(RuntimeError, match="SEC request failed"):
            ingest_and_chunk_filing(
                cik="0000320193",
                form_type="10-K",
                user_agent="test@example.com",
            )
