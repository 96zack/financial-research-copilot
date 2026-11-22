import pytest

from financial_research_copilot.evaluation.grounding import (
    ClaimEvaluation,
    GroundingEvaluation,
    abstention_correct,
    grounding_score,
)


def make_claim(supported: bool) -> ClaimEvaluation:
    """Create a sample claim for testing."""
    return ClaimEvaluation(
        claim="Product mix affected Apple's gross margin.",
        cited_sources=[1],
        supported=supported,
        explanation="Test evidence assessment.",
    )


def test_grounding_score_all_supported():
    evaluation = GroundingEvaluation(
        claims=[
            make_claim(True),
            make_claim(True),
            make_claim(True),
        ]
    )

    assert grounding_score(evaluation) == 1.0


def test_grounding_score_partially_supported():
    evaluation = GroundingEvaluation(
        claims=[
            make_claim(True),
            make_claim(False),
            make_claim(True),
        ]
    )

    assert grounding_score(evaluation) == pytest.approx(2 / 3)


def test_grounding_score_empty_claims():
    evaluation = GroundingEvaluation(claims=[])

    assert grounding_score(evaluation) == 1.0


def test_abstention_correct():
    assert abstention_correct(True, True)
    assert abstention_correct(False, False)


def test_abstention_incorrect():
    assert not abstention_correct(True, False)
    assert not abstention_correct(False, True)
