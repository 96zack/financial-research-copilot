from pydantic import BaseModel, Field


class StructuredFinancialAnswer(BaseModel):
    answer: str = Field(
        description="A concise answer grounded in the provided SEC filing excerpts."
    )

    cited_sources: list[int] = Field(
        description="Source numbers supporting the answer, such as [1, 2]."
    )

    insufficient_evidence: bool = Field(
        description="True if the supplied excerpts cannot adequately answer the question."
    )
