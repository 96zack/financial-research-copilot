from financial_research_copilot.processing.html_parser import (
    remove_page_footers,
)


def test_remove_page_footers():
    text = (
        "Supply chain risks\n"
        "Apple Inc. | 2026 Form 10-K | 3\n"
        "Manufacturing concentration creates risks."
    )

    result = remove_page_footers(text)

    assert result == ("Supply chain risks\nManufacturing concentration creates risks.")


def test_preserve_normal_text():
    text = (
        "Apple Inc. faces supply chain risks.\n"
        "The company depends on third-party manufacturers."
    )

    assert remove_page_footers(text) == text


def test_preserve_heading_and_disclosure():
    text = (
        "Supply of Components and Computing Resources\n"
        "Although most components are available from multiple sources."
    )

    assert remove_page_footers(text) == text
