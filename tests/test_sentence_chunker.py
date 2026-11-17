import pytest

from financial_research_copilot.processing.sentence_chunker import (
    split_into_sentences,
    split_long_paragraph,
    split_long_sentence,
)


def test_split_into_sentences():
    text = "Apple sells iPhones. It also offers services."
    assert split_into_sentences(text) == [
        "Apple sells iPhones.",
        "It also offers services.",
    ]


def test_sentence_boundaries():
    text = (
        "Apple relies on suppliers. "
        "Supply shortages increase costs. "
        "Margins may decline."
    )

    assert split_long_paragraph(text, 55) == [
        "Apple relies on suppliers.",
        "Supply shortages increase costs. Margins may decline.",
    ]


def test_long_sentence():
    result = split_long_paragraph("Apple " + "A" * 30, 10)

    assert all(len(chunk) <= 10 for chunk in result)
    assert result == ["Apple", "AAAAAAAAAA", "AAAAAAAAAA", "AAAAAAAAAA"]


def test_empty_text():
    assert split_long_paragraph("", 100) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_long_paragraph("Apple.", 0)


def test_long_word_preserves_characters():
    result = split_long_sentence("ABCDEFGHIJ", 4)
    assert result == ["ABCD", "EFGH", "IJ"]


def test_no_empty_chunks():
    result = split_long_paragraph("A" * 50, 20)
    assert all(result)
