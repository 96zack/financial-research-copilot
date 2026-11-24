from unittest.mock import MagicMock, patch

from financial_research_copilot.evaluation.grounding import (
    ClaimEvaluation,
    GroundingEvaluation,
)
from financial_research_copilot.evaluation.grounding_judge import (
    evaluate_grounding,
)


def test_evaluate_grounding():
    expected = GroundingEvaluation(
        claims=[
            ClaimEvaluation(
                claim="Product mix affected margins.",
                cited_sources=[1],
                supported=True,
                explanation="Source 1 supports the claim.",
            )
        ]
    )

    mock_response = MagicMock()
    mock_response.output_parsed = expected

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch(
            "financial_research_copilot.evaluation.grounding_judge.OpenAI"
        ) as mock_openai,
    ):
        mock_openai.return_value.responses.parse.return_value = mock_response

        result = evaluate_grounding(
            answer="Product mix affected margins [Source 1].",
            context="[Source 1]\nProduct mix affected margins.",
        )

    # TODO 1: Assert the returned result equals expected.
    assert result == expected

    # TODO 2: Assert responses.parse() was called exactly once.
    mock_openai.return_value.responses.parse.assert_called_once()
