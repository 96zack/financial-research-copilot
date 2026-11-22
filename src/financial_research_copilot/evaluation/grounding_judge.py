import os

from dotenv import load_dotenv
from openai import OpenAI

from financial_research_copilot.evaluation.grounding import (
    GroundingEvaluation,
)

load_dotenv()

GROUNDING_JUDGE_INSTRUCTIONS = """
You are evaluating whether a financial research answer is
supported by the provided SEC filing excerpts.

1. Break the answer into individual factual claims.
2. Identify the source numbers cited for each claim.
3. Evaluate each claim using ONLY its cited sources.
4. Mark supported=True only when the cited excerpts
   adequately support the claim.
5. Mark unsupported predictions, fabricated numbers,
   and contradictions as supported=False.
6. Provide a brief explanation for every judgment.
7. Do not use external knowledge.
"""


def evaluate_grounding(
    answer: str,
    context: str,
) -> GroundingEvaluation:
    if not answer.strip() or not context.strip():
        raise ValueError("Answer and context cannot be empty")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    response = client.responses.parse(
        model=model,
        instructions=GROUNDING_JUDGE_INSTRUCTIONS,
        input=f"Context:\n{context}\n\nAnswer:\n{answer}",
        text_format=GroundingEvaluation,
    )

    if response.output_parsed is None:
        raise ValueError("Judge did not return an evaluation")

    return response.output_parsed
