import pytest

from financial_research_copilot.processing.chunker import (
    split_text,
    split_text_by_paragraph,
)


def test_split_text_with_overlap():
    result = split_text(
        "ABCDEFGHIJ",
        chunk_size=6,
        chunk_overlap=2,
    )

    assert result == ["ABCDEF", "EFGHIJ"]


def test_split_text_short_input():
    result = split_text(
        "Hello",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert result == ["Hello"]


def test_split_text_empty_input():
    assert split_text("") == []


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_split_text_invalid_parameters(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        split_text(
            "Example",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_paragraph_chunking_basic():
    text = "AAAA\n\nBBBB\n\nCCCC"

    result = split_text_by_paragraph(
        text,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert result == ["AAAA\n\nBBBB", "CCCC"]


def test_paragraph_chunking_oversized():
    text = "AAAA\n\nBBBBBBBBBBBB\n\nCCCC"

    result = split_text_by_paragraph(
        text,
        chunk_size=8,
        chunk_overlap=2,
    )

    assert result == ["AAAA", "BBBBBBBB", "BBBBBB", "CCCC"]


def test_paragraph_chunking_empty():
    assert split_text_by_paragraph("") == []
    assert split_text_by_paragraph("\n\n  \n") == []


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_paragraph_chunking_invalid_parameters(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        split_text_by_paragraph(
            "Example",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_paragraph_chunks_respect_max_size():
    text = "A" * 1500 + "\n\n" + "B" * 300

    chunks = split_text_by_paragraph(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )
    assert all(0 < len(chunk) <= 1000 for chunk in chunks)
