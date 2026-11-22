from pydantic import BaseModel, Field


class ClaimEvaluation(BaseModel):
    claim: str = Field(description="A single factual claim extracted from the answer.")
    cited_sources: list[int] = Field(description="Source numbers cited for this claim.")
    supported: bool = Field(
        description="Whether the cited evidence supports this claim."
    )
    explanation: str = Field(
        description="Brief justification for the support decision."
    )


class GroundingEvaluation(BaseModel):
    claims: list[ClaimEvaluation]


def grounding_score(evaluation: GroundingEvaluation) -> float:
    """Fraction of evaluated claims supported by cited evidence."""

    # TODO 1: Handle an empty claims list.
    # Return 1.0 for now.
    if not evaluation.claims:
        return 1.0

    # TODO 2: Count claims where supported is True.
    support_count = 0

    for claim in evaluation.claims:
        if claim.supported:
            support_count += 1

    # TODO 3: Divide supported claims by total claims.
    return support_count / len(evaluation.claims)


def abstention_correct(
    predicted_insufficient_evidence: bool,
    expected_abstention: bool,
) -> bool:
    """Check whether the model made the correct abstention decision."""
    return predicted_insufficient_evidence == expected_abstention
