from unittest.mock import patch

import pytest

from financial_research_copilot.generation.rag_pipeline import (
    answer_question,
    answer_question_with_citations,
    format_context,
)
from tests.test_citations import make_chunk


def test_format_context():
    chunks = [
        "Apple relies on suppliers.",
        "Component shortages may increase costs.",
    ]

    result = format_context(chunks)

    assert result == (
        "[Source 1]\nApple relies on suppliers.\n\n"
        "[Source 2]\nComponent shortages may increase costs."
    )


def test_format_context_empty():
    assert format_context([]) == ""


def test_answer_question_empty_chunks():
    with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
        answer_question("What are Apple's risks?", [])


def test_answer_question_calls_llm():
    chunks = [
        "Apple relies on suppliers.",
        "Component shortages may increase costs.",
    ]

    with patch(
        "financial_research_copilot.generation.rag_pipeline.generate_answer",
        return_value="Supply shortages could increase costs.",
    ) as mock_generate:
        result = answer_question(
            "What are Apple's supply risks?",
            chunks,
        )

    assert result == "Supply shortages could increase costs."

    mock_generate.assert_called_once_with(
        "What are Apple's supply risks?",
        (
            "[Source 1]\nApple relies on suppliers.\n\n"
            "[Source 2]\nComponent shortages may increase costs."
        ),
    )


def test_answer_question_with_valid_citations():
    chunks = [
        make_chunk("chunk_001", "Apple relies on suppliers."),
        make_chunk("chunk_002", "Component costs affect margins."),
    ]

    with patch(
        "financial_research_copilot.generation.rag_pipeline.generate_answer",
        return_value="Supplier risks [Source 1] and cost risks [Source 2].",
    ) as mock_generate:
        result = answer_question_with_citations(
            "What risks affect Apple's margins?",
            chunks,
        )

    assert result.citations_valid is True
    assert result.answer == "Supplier risks [Source 1] and cost risks [Source 2]."
    assert len(result.sources) == 2
    assert result.sources[0].chunk_id == "chunk_001"
    assert result.sources[1].chunk_id == "chunk_002"

    mock_generate.assert_called_once_with(
        "What risks affect Apple's margins?",
        (
            "[Source 1]\nApple relies on suppliers.\n\n"
            "[Source 2]\nComponent costs affect margins."
        ),
    )


def test_answer_question_with_invalid_citations():
    chunks = [
        make_chunk("chunk_001", "Apple relies on suppliers."),
    ]

    with patch(
        "financial_research_copilot.generation.rag_pipeline.generate_answer",
        return_value="Apple faces supplier risks [Source 5].",
    ):
        result = answer_question_with_citations(
            "What are Apple's risks?",
            chunks,
        )

    assert result.citations_valid is False
    assert len(result.sources) == 1
