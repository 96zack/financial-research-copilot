from dataclasses import dataclass

from financial_research_copilot.generation.citations import (
    CitationSource,
    build_citation_sources,
    validate_citations,
)
from financial_research_copilot.generation.llm_client import generate_answer
from financial_research_copilot.models.chunk import DocumentChunk


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[CitationSource]
    citations_valid: bool


def format_context(chunks: list[str]) -> str:
    """Format retrieved passages into numbered evidence blocks."""
    formatted_chunks = []
    for ind, text in enumerate(chunks, start=1):
        formatted_chunks.append(f"[Source {ind}]\n{text}")
    return "\n\n".join(formatted_chunks)


def answer_question(
    question: str,
    retrieved_chunks: list[str],
) -> str:
    """Generate an answer using retrieved SEC filing passages."""

    # TODO 1: Validate retrieved_chunks is not empty
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")

    # TODO 2: Call format_context(retrieved_chunks)
    formatted_context = format_context(retrieved_chunks)

    # TODO 3: Call generate_answer(question, context)
    answer = generate_answer(question, formatted_context)

    # TODO 4: Return the generated answer
    return answer


def answer_question_with_citations(
    question: str,
    retrieved_chunks: list[DocumentChunk],
) -> RAGResponse:
    """Generate an answer with source references."""

    # TODO 1: Validate that retrieved_chunks is not empty
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")
    # TODO 2: Build numbered citation sources
    citation_sources = build_citation_sources(retrieved_chunks)

    # TODO 3: Format the sources into context
    context = format_context([source.text for source in citation_sources])

    # TODO 4: Generate the answer
    answer = generate_answer(question, context)

    # TODO 5: Validate the citations
    validation_result = validate_citations(answer, citation_sources)

    # TODO 6: Return RAGResponse
    return RAGResponse(
        answer=answer, sources=citation_sources, citations_valid=validation_result
    )
