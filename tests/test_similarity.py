import pytest

from financial_research_copilot.retrieval.similarity import (
    cosine_similarity,
)


def test_identical_direction():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_opposite_direction():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_diff_mag():
    assert cosine_similarity([3, 4], [6, 8]) == pytest.approx(1.0)


def test_nontrivial_similarity():
    assert cosine_similarity([1, 2], [2, 1]) == pytest.approx(0.8)


def test_different_length():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1])


def test_empty_vec():
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_one_zero_vec():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])


def test_two_zero_vec():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [0, 0])
