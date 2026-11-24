import re
from dataclasses import dataclass

from financial_research_copilot.generation.citations import (
    CitationSource,
    build_citation_sources,
    validate_citations,
)
from financial_research_copilot.generation.llm_client import (
    generate_answer,
    generate_structured_answer,
)
from financial_research_copilot.models.chunk import DocumentChunk
from financial_research_copilot.models.rag_response import StructuredFinancialAnswer


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[CitationSource]
    citations_valid: bool


@dataclass(frozen=True)
class StructuredRAGResponse:
    result: StructuredFinancialAnswer
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


def answer_question_structured(
    question: str,
    retrieved_chunks: list[DocumentChunk],
) -> StructuredRAGResponse:
    """Generate a structured, citation-validated RAG answer."""

    # TODO 1: Reject empty retrieved_chunks
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")
    # TODO 2: Build CitationSource objects
    citation_sources = build_citation_sources(retrieved_chunks)

    # TODO 3: Format context from source texts
    context = format_context([source.text for source in citation_sources])

    # TODO 4: Call generate_structured_answer(question, context)
    structured_answer = generate_structured_answer(question, context)

    # TODO 5: Validate result.cited_sources against available source numbers
    available_numbers = {source.source_number for source in citation_sources}

    listed_numbers = set(structured_answer.cited_sources)

    inline_numbers = {
        int(number)
        for number in re.findall(r"\[Source (\d+)\]", structured_answer.answer)
    }

    citations_valid = (
        listed_numbers.issubset(available_numbers)
        and inline_numbers.issubset(available_numbers)
        and listed_numbers == inline_numbers
    )

    # TODO 6: Return StructuredRAGResponse
    return StructuredRAGResponse(
        result=structured_answer,
        sources=citation_sources,
        citations_valid=citations_valid,
    )
