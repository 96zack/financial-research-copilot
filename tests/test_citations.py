from financial_research_copilot.generation.citations import (
    CitationSource,
    build_citation_sources,
    validate_citations,
)
from financial_research_copilot.models.chunk import (
    ChunkMetadata,
    DocumentChunk,
)


def make_source(number: int) -> CitationSource:
    return CitationSource(
        source_number=number,
        chunk_id=f"chunk_{number}",
        text=f"Evidence from source {number}.",
    )


def test_validate_citations_valid():
    sources = [make_source(1), make_source(2), make_source(3)]

    answer = "Apple faces supply risks [Source 1] and cost pressures [Source 3]."

    assert validate_citations(answer, sources) is True


def test_validate_citations_invalid():
    sources = [make_source(1), make_source(2), make_source(3)]

    answer = "Apple faces financial risks [Source 5]."

    assert validate_citations(answer, sources) is False


def test_validate_citations_no_references():
    sources = [make_source(1), make_source(2), make_source(3)]

    answer = "Apple faces financial risks."

    assert validate_citations(answer, sources) is True


def test_validate_citations_repeated_reference():
    sources = [make_source(1), make_source(2), make_source(3)]

    answer = "Apple faces supply risks [Source 1] and cost pressures [Source 1]."

    assert validate_citations(answer, sources) is True


def test_validate_citations_empty_sources():
    sources = []

    answer = "Apple faces supply risks [Source 1] "

    assert validate_citations(answer, sources) is False


def make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            cik="0000320193",
            company_name="Apple Inc.",
            form="10-K",
            filing_date="2026-10-30",
            source_url="https://www.sec.gov/",
            chunk_index=0,
        ),
    )


def test_build_citation_sources_preserves_order():
    chunks = [
        make_chunk("chunk_025", "Apple relies on suppliers."),
        make_chunk("chunk_102", "Component costs affect margins."),
    ]

    sources = build_citation_sources(chunks)

    assert len(sources) == 2
    assert [s.source_number for s in sources] == [1, 2]
    assert [s.chunk_id for s in sources] == [
        "chunk_025",
        "chunk_102",
    ]
    assert [s.text for s in sources] == [
        "Apple relies on suppliers.",
        "Component costs affect margins.",
    ]


def test_build_citation_sources_empty():
    assert build_citation_sources([]) == []
