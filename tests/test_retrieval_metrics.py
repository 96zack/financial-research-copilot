import pytest

from financial_research_copilot.retrieval.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.fixture
def sample_data():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "c", "e", "f"}
    return retrieved, relevant


def test_precision_at_k(sample_data):
    retrieved, relevant = sample_data
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k(sample_data):
    retrieved, relevant = sample_data
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)


def test_reciprocal_rank(sample_data):
    retrieved, relevant = sample_data
    assert reciprocal_rank(retrieved, relevant, 3) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant():
    assert reciprocal_rank(["a", "b", "c"], {"d"}, 3) == 0.0


def test_reciprocal_rank_beyond_k():
    assert reciprocal_rank(["a", "b", "c"], {"c"}, 2) == 0.0


@pytest.mark.parametrize(
    "metric",
    [precision_at_k, recall_at_k, reciprocal_rank],
)
def test_invalid_k(metric):
    with pytest.raises(ValueError):
        metric(["a", "b"], {"a"}, 0)


@pytest.mark.parametrize(
    "metric",
    [precision_at_k, recall_at_k, reciprocal_rank],
)
def test_duplicate_ids(metric):
    with pytest.raises(ValueError):
        metric(["a", "a", "b"], {"a"}, 3)


def test_mean_reciprocal_rank():
    scores = [1.0, 0.5, 0.25]
    assert mean_reciprocal_rank(scores) == pytest.approx(7 / 12)


def test_empty_mean_reciprocal_rank():
    assert mean_reciprocal_rank([]) == 0.0


def test_recall_empty_relevant_set():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_precision_with_fewer_results_than_k():
    assert precision_at_k(["a"], {"a"}, 3) == pytest.approx(1 / 3)
