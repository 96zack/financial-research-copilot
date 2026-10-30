from unittest.mock import Mock, patch

import httpx
import pytest

from financial_research_copilot.ingestion.sec_client import (
    download_filing,
    get_company_submissions,
    get_latest_filing,
)
from financial_research_copilot.processing.html_parser import (
    extract_filing_text,
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


@patch("financial_research_copilot.ingestion.sec_client.httpx.get")
def test_download_filing_success(mock_get):
    fake_response = Mock()
    fake_response.text = "<html><body>Apple 10-K</body></html>"
    mock_get.return_value = fake_response

    url = "https://www.sec.gov/Archives/edgar/data/example.htm"

    result = download_filing(
        url=url,
        user_agent="TestApp test@example.com",
    )

    assert result == "<html><body>Apple 10-K</body></html>"

    mock_get.assert_called_once_with(
        url,
        headers={"User-Agent": "TestApp test@example.com"},
        timeout=30.0,
        follow_redirects=True,
    )


@patch("financial_research_copilot.ingestion.sec_client.httpx.get")
def test_download_filing_http_error(mock_get):
    url = "https://www.sec.gov/Archives/edgar/data/example.htm"

    request = httpx.Request("GET", url)
    response = httpx.Response(status_code=403, request=request)

    mock_get.return_value = response

    # TODO: Verify download_filing() raises HTTPStatusError
    with pytest.raises(httpx.HTTPStatusError):
        download_filing(
            url=url,
            user_agent="TestApp test@example.com",
        )


def test_remove_xbrl_header_preserve_visible_facts():
    html = """
    <html>
        <body>
            <ix:header>
                <ix:hidden>
                    <ix:nonfraction>999999</ix:nonfraction>
                </ix:hidden>
                <ix:resources>
                    http://fasb.org/us-gaap/2025
                </ix:resources>
            </ix:header>

            <p>
                Net sales were
                <ix:nonfraction>391,035</ix:nonfraction>
                million.
            </p>
        </body>
    </html>
    """

    result = extract_filing_text(html)

    assert "999999" not in result
    assert "http://fasb.org" not in result
    assert "391,035" in result


def test_preserve_inline_text():
    html = """
    <p>
        For the fiscal year ended
        <span>September 27</span>, 2025
    </p>
    """

    result = extract_filing_text(html)

    assert "September 27" in result
    assert "2025" in result
