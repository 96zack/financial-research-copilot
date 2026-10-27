from datetime import date

import httpx

SEC_BASE_URL = "https://data.sec.gov/submissions"


def get_company_submissions(cik: str, user_agent: str) -> dict:
    padded_cik = cik.zfill(10)
    url = f"{SEC_BASE_URL}/CIK{padded_cik}.json"

    headers = {"User-Agent": user_agent}

    response = httpx.get(
        url,
        headers=headers,
        timeout=15.0,
    )
    response.raise_for_status()

    return response.json()


def get_latest_filing(submissions: dict, form_type: str) -> dict:
    recent = submissions["filings"]["recent"]
    forms = recent["form"]
    latest_date = None
    latest_date_index = None

    for index, form in enumerate(forms):
        # TODO: Check whether form matches form_type.
        if form == form_type:
            file_date = date.fromisoformat(recent["filingDate"][index])

            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_date_index = index

    if latest_date_index is None:
        raise ValueError(f"No {form_type} filing found")

    # TODO: Extract the filing's other fields using index.
    result = {
        "form": recent["form"][latest_date_index],
        "filing_date": recent["filingDate"][latest_date_index],
        "accession_number": recent["accessionNumber"][latest_date_index],
        "primary_document": recent["primaryDocument"][latest_date_index],
    }

    return result
