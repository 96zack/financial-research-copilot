import pytest

from financial_research_copilot.evaluation.citation_metrics import (
    citation_coverage,
    citation_precision,
)


def test_citation_coverage_partial():
    answer = (
        "Product mix affected margins [Source 1]. "
        "Costs increased. "
        "Services margins improved [Source 2]."
    )
    assert citation_coverage(answer) == pytest.approx(2 / 3)


def test_citation_coverage_full():
    answer = "Margins increased [Source 1]. Costs increased [Source 2]."
    assert citation_coverage(answer) == 1.0


def test_citation_coverage_empty():
    assert citation_coverage("") == 1.0


def test_citation_precision_partial():
    answer = (
        "Margins increased [Source 1]. "
        "Costs increased [Source 4]. "
        "Services improved [Source 1]."
    )
    assert citation_precision(answer, {1, 2, 3}) == pytest.approx(2 / 3)


def test_citation_precision_full():
    answer = "Margins increased [Source 1] [Source 2]."
    assert citation_precision(answer, {1, 2}) == 1.0


def test_citation_precision_no_citations():
    assert citation_precision("Margins increased.", {1, 2}) == 1.0
