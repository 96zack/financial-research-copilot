from financial_research_copilot.ingestion.sec_client import (
    build_filing_url,
    download_filing,
    get_company_submissions,
    get_latest_filing,
)
from financial_research_copilot.processing.html_parser import (
    extract_filing_text,
)


def ingest_latest_filing(
    cik: str,
    form_type: str,
    user_agent: str,
) -> dict:
    # Step 1: Retrieve submissions
    submissions = get_company_submissions(cik, user_agent)

    # Step 2: Find latest filing
    filing = get_latest_filing(submissions, form_type)

    # Step 3: Build URL
    url = build_filing_url(
        cik=cik,
        accession_number=filing["accession_number"],
        primary_document=filing["primary_document"],
    )

    # Step 4: Download filing HTML
    html = download_filing(url, user_agent)

    # Step 5: Extract text
    text = extract_filing_text(html)

    # Step 6: Return metadata and text
    return {
        "company_name": submissions["name"],
        "cik": str(int(submissions["cik"])),
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "url": url,
        "text": text,
    }
