import re
from dataclasses import dataclass

from financial_research_copilot.models.chunk import DocumentChunk


@dataclass(frozen=True)
class CitationSource:
    source_number: int
    chunk_id: str
    text: str


def build_citation_sources(
    chunks: list[DocumentChunk],
) -> list[CitationSource]:
    # TODO: Implement
    result = []
    for ind, chunk in enumerate(chunks, start=1):
        result.append(
            CitationSource(source_number=ind, chunk_id=chunk.chunk_id, text=chunk.text)
        )
    return result


def validate_citations(
    answer: str,
    sources: list[CitationSource],
) -> bool:
    """
    Return True if all referenced source numbers exist.
    """
    # TODO 1: Extract citation numbers from answer
    referenced = set(re.findall(r"\[Source (\d+)\]", answer))
    # TODO 2: Build a set of valid source numbers
    valid_numbers = {str(source.source_number) for source in sources}
    # TODO 3: Check whether every citation number is valid
    # TODO 4: Return the result
    return referenced.issubset(valid_numbers)
