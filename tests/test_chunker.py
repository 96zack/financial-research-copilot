
import pytest

from financial_research_copilot.processing.chunker import split_text


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
