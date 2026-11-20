from unittest.mock import MagicMock, patch

from financial_research_copilot.generation.llm_client import (
    generate_structured_answer,
)
from financial_research_copilot.models.rag_response import (
    StructuredFinancialAnswer,
)


def test_generate_structured_answer():
    expected = StructuredFinancialAnswer(
        answer="Apple's gross margin was affected by product mix [Source 1].",
        cited_sources=[1],
        insufficient_evidence=False,
    )

    mock_response = MagicMock()
    mock_response.output_parsed = expected

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("financial_research_copilot.generation.llm_client.OpenAI") as mock_openai,
    ):
        mock_openai.return_value.responses.parse.return_value = mock_response

        result = generate_structured_answer(
            question="What affected Apple's gross margin?",
            context="[Source 1]\nProduct mix affected gross margin.",
        )

    assert result == expected
    mock_openai.return_value.responses.parse.assert_called_once()
