import pytest

from financial_research_copilot.retrieval.search import top_k_search


def test_top_k_search_basic():
    query = [1.0, 0.0]
    documents = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
    ]

    results = top_k_search(query, documents, k=2)

    assert len(results) == 2
    assert [index for index, _ in results] == [0, 2]
    assert [score for _, score in results] == pytest.approx([1.0, 0.8])


def test_top_k_search_k_exceeds_documents():
    query = [1.0, 0.0]
    documents = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    results = top_k_search(query, documents, k=5)

    assert len(results) == 2
    assert [index for index, _ in results] == [0, 1]


def test_top_k_search_invalid_k():
    query = [1.0, 0.0]
    documents = [[1.0, 0.0]]

    with pytest.raises(ValueError):
        top_k_search(query, documents, k=0)


def test_top_k_search_empty_documents():
    query = [1.0, 0.0]

    results = top_k_search(query, [], k=3)

    assert results == []


def test_top_k_search_equal_scores():
    query = [1.0, 0.0]
    documents = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    results = top_k_search(query, documents, k=2)

    assert [index for index, _ in results] == [0, 1]
    assert [score for _, score in results] == pytest.approx([1.0, 1.0])


def test_top_k_search_dimension_mismatch():
    query = [1.0, 0.0]
    documents = [
        [1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    with pytest.raises(ValueError):
        top_k_search(query, documents, k=2)


def test_top_k_search_negative_k():
    query = [1.0, 0.0]
    documents = [[1.0, 0.0]]

    with pytest.raises(ValueError):
        top_k_search(query, documents, k=-1)
