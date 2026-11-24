import pytest
from pydantic import ValidationError

from financial_research_copilot.models.rag_response import (
    StructuredFinancialAnswer,
)


def test_structured_financial_answer_valid():
    response = StructuredFinancialAnswer(
        answer="Apple's gross margin was affected by product mix [Source 1].",
        cited_sources=[1, 2],
        insufficient_evidence=False,
    )

    assert response.answer == (
        "Apple's gross margin was affected by product mix [Source 1]."
    )
    assert response.cited_sources == [1, 2]
    assert response.insufficient_evidence is False


def test_structured_financial_answer_invalid_citation_type():
    with pytest.raises(ValidationError):
        StructuredFinancialAnswer(
            answer="Apple's gross margin increased [Source 1].",
            cited_sources=["Source 1"],
            insufficient_evidence=False,
        )


def test_structured_financial_answer_missing_field():
    with pytest.raises(ValidationError):
        StructuredFinancialAnswer(
            answer="Insufficient information.",
            cited_sources=[],
        )
