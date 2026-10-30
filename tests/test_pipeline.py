from unittest.mock import patch

import httpx
import pytest

from financial_research_copilot.ingestion.pipeline import (
    ingest_latest_filing,
)

SAMPLE_SUBMISSIONS = {
    "name": "Apple Inc.",
    "cik": 320193,
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K"],
            "filingDate": ["2026-01-30", "2025-10-31"],
            "accessionNumber": [
                "0000320193-26-000001",
                "0000320193-25-000079",
            ],
            "primaryDocument": [
                "aapl-20251227.htm",
                "aapl-20250927.htm",
            ],
        }
    },
}

SAMPLE_HTML = """
<html>
    <body>
        <ix:header>
            <ix:resources>HIDDEN_XBRL_METADATA</ix:resources>
        </ix:header>
        <h2>Item 1A. Risk Factors</h2>
        <p>Apple faces supply chain risks.</p>
    </body>
</html>
"""


@patch("financial_research_copilot.ingestion.pipeline.download_filing")
@patch("financial_research_copilot.ingestion.pipeline.get_company_submissions")
def test_ingest_latest_filing(mock_submissions, mock_download):
    mock_submissions.return_value = SAMPLE_SUBMISSIONS
    mock_download.return_value = SAMPLE_HTML

    result = ingest_latest_filing(
        cik="0000320193",
        form_type="10-K",
        user_agent="Test Agent test@example.com",
    )

    # TODO: Write your assertions here
    assert result["company_name"] == "Apple Inc."
    assert result["cik"] == "320193"
    assert result["form"] == "10-K"
    assert result["filing_date"] == "2025-10-31"
    assert "Apple faces supply chain risks." in result["text"]
    assert "HIDDEN_XBRL_METADATA" not in result["text"]

    mock_submissions.assert_called_once_with(
        "0000320193",
        "Test Agent test@example.com",
    )
    mock_download.assert_called_once_with(
        "https://www.sec.gov/Archives/edgar/data/"
        "320193/000032019325000079/aapl-20250927.htm",
        "Test Agent test@example.com",
    )


@patch("financial_research_copilot.ingestion.pipeline.download_filing")
@patch("financial_research_copilot.ingestion.pipeline.get_company_submissions")
def test_ingest_latest_filing_http_error(mock_submissions, mock_download):
    mock_submissions.return_value = SAMPLE_SUBMISSIONS

    request = httpx.Request(
        "GET",
        "https://www.sec.gov/Archives/edgar/data/example.htm",
    )

    response = httpx.Response(
        status_code=403,
        request=request,
    )

    mock_download.side_effect = httpx.HTTPStatusError(
        "403 Forbidden",
        request=request,
        response=response,
    )

    with pytest.raises(httpx.HTTPStatusError):
        ingest_latest_filing(
            cik="320193",
            form_type="10-K",
            user_agent="Test Agent test@example.com",
        )
