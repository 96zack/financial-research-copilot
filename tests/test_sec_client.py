from unittest.mock import Mock, patch

import httpx
import pytest

from financial_research_copilot.ingestion.sec_client import (
    get_company_submissions,
    get_latest_filing,
)

SAMPLE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "10-K"],
            "filingDate": [
                "2024-11-01",
                "2025-08-01",
                "2025-11-01",
            ],
            "accessionNumber": ["A001", "A002", "A003"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"],
        }
    }
}


def test_get_latest_10k():
    result = get_latest_filing(SAMPLE_SUBMISSIONS, "10-K")

    assert result["filing_date"] == "2025-11-01"
    assert result["accession_number"] == "A003"


def test_get_latest_10q():
    result = get_latest_filing(SAMPLE_SUBMISSIONS, "10-Q")

    assert result["filing_date"] == "2025-08-01"
    assert result["accession_number"] == "A002"


def test_no_matching_filing():
    with pytest.raises(ValueError):
        get_latest_filing(SAMPLE_SUBMISSIONS, "NOT-A-REAL-FORM")


@patch("financial_research_copilot.ingestion.sec_client.httpx.get")
def test_get_company_submissions(mock_get):
    fake_response = Mock()
    fake_response.json.return_value = {
        "name": "Apple Inc.",
        "cik": 320193,
    }
    fake_response.raise_for_status.return_value = None

    mock_get.return_value = fake_response

    result = get_company_submissions(
        cik="320193",
        user_agent="TestApp test@example.com",
    )

    assert result["name"] == "Apple Inc."
    assert result["cik"] == 320193

    mock_get.assert_called_once_with(
        "https://data.sec.gov/submissions/CIK0000320193.json",
        headers={"User-Agent": "TestApp test@example.com"},
        timeout=15.0,
    )


@patch("financial_research_copilot.ingestion.sec_client.httpx.get")
def test_get_company_submissions_http_error(mock_get):
    request = httpx.Request(
        "GET",
        "https://data.sec.gov/submissions/CIK0000320193.json",
    )

    response = httpx.Response(
        status_code=404,
        request=request,
    )

    mock_get.return_value = response

    with pytest.raises(httpx.HTTPStatusError):
        get_company_submissions(
            cik="320193",
            user_agent="TestApp test@example.com",
        )
