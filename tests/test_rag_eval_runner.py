from unittest.mock import patch

import pytest

from experiments.rag_eval_dataset import RAGEvalCase
from experiments.rag_eval_runner import (
    RAGEvalResult,
    evaluate_case,
    summarize_results,
)
from financial_research_copilot.evaluation.grounding import (
    ClaimEvaluation,
    GroundingEvaluation,
)
from financial_research_copilot.generation.citations import CitationSource
from financial_research_copilot.generation.rag_pipeline import (
    StructuredRAGResponse,
)
from financial_research_copilot.models.rag_response import (
    StructuredFinancialAnswer,
)
from tests.factories import make_chunk


def make_result(
    coverage: float = 1.0,
    precision: float = 1.0,
    grounding: float = 1.0,
    abstention: bool = True,
    validity: bool = True,
) -> RAGEvalResult:
    return RAGEvalResult(
        case_id="test_case",
        question="Test question",
        answer="Test answer",
        citation_coverage=coverage,
        citation_precision=precision,
        grounding_score=grounding,
        citations_valid=validity,
        insufficient_evidence=False,
        abstention_correct=abstention,
    )


def test_summarize_results_perfect():
    results = [make_result(), make_result()]

    summary = summarize_results(results)

    assert summary == {
        "citation_coverage": 1.0,
        "citation_precision": 1.0,
        "grounding_score": 1.0,
        "abstention_accuracy": 1.0,
        "citation_validity_rate": 1.0,
    }


def test_summarize_results_mixed():
    results = [
        make_result(1.0, 1.0, 1.0, True, True),
        make_result(0.5, 0.5, 0.8, False, False),
        make_result(0.0, 1.0, 0.6, True, False),
    ]

    summary = summarize_results(results)

    assert summary["citation_coverage"] == pytest.approx(0.5)
    assert summary["citation_precision"] == pytest.approx(2.5 / 3)
    assert summary["grounding_score"] == pytest.approx(0.8)
    assert summary["abstention_accuracy"] == pytest.approx(2 / 3)
    assert summary["citation_validity_rate"] == pytest.approx(1 / 3)


def test_summarize_results_empty():
    with pytest.raises(ValueError, match="results cannot be empty"):
        summarize_results([])


def test_evaluate_case():
    case = RAGEvalCase(
        case_id="gross_margin",
        question="What affected Apple's gross margin?",
        expected_source_topics=("product mix",),
        should_abstain=False,
    )

    source_text = "Product mix contributed to higher gross margin."
    chunks = [make_chunk("chunk-1", source_text)]

    mock_rag_response = StructuredRAGResponse(
        result=StructuredFinancialAnswer(
            answer="Product mix affected margins [Source 1].",
            cited_sources=[1],
            insufficient_evidence=False,
        ),
        sources=[
            CitationSource(
                source_number=1,
                chunk_id="chunk-1",
                text=source_text,
            )
        ],
        citations_valid=True,
    )

    mock_grounding = GroundingEvaluation(
        claims=[
            ClaimEvaluation(
                claim="Product mix affected margins.",
                cited_sources=[1],
                supported=True,
                explanation="Source 1 supports the claim.",
            )
        ]
    )

    with (
        patch(
            "experiments.rag_eval_runner.answer_question_structured",
            return_value=mock_rag_response,
        ) as mock_generate,
        patch(
            "experiments.rag_eval_runner.evaluate_grounding",
            return_value=mock_grounding,
        ) as mock_judge,
    ):
        result = evaluate_case(case, chunks)

    assert result.case_id == case.case_id
    assert result.question == case.question
    assert result.answer == mock_rag_response.result.answer

    assert result.citation_coverage == 1.0
    assert result.citation_precision == 1.0
    assert result.grounding_score == 1.0

    assert result.citations_valid is True
    assert result.insufficient_evidence is False
    assert result.abstention_correct is True

    mock_generate.assert_called_once_with(
        question=case.question,
        retrieved_chunks=chunks,
    )

    mock_judge.assert_called_once_with(
        answer=mock_rag_response.result.answer,
        context=f"[Source 1]\n{source_text}",
    )
