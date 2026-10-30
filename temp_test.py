from financial_research_copilot.ingestion.sec_client import (
    build_filing_url,
    download_filing,
    get_company_submissions,
    get_latest_filing,
)
from financial_research_copilot.processing.html_parser import (
    extract_filing_text,
)

USER_AGENT = "FinancialResearchCopilot zackchen96@gmail.com"

submissions = get_company_submissions("320193", USER_AGENT)
filing = get_latest_filing(submissions, "10-K")

url = build_filing_url(
    cik="320193",
    accession_number=filing["accession_number"],
    primary_document=filing["primary_document"],
)

html = download_filing(url, USER_AGENT)
text = extract_filing_text(html)

print("Extracted text characters:", len(text))
print("First 1000 characters:")
print(text[:1000])

print("Risk Factors present:", "Risk Factors" in text)
print("Business section present:", "Business" in text)
print("XBRL taxonomy noise:", "http://fasb.org/us-gaap" in text)
