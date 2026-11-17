from unittest.mock import patch

import pytest

from financial_research_copilot.generation.rag_pipeline import (
    answer_question,
    format_context,
)


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
