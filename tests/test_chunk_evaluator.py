import pytest

from financial_research_copilot.processing.chunk_evaluator import (
    evaluate_chunks,
)


def test_evaluate_chunks_basic():
    chunks = [
        "A" * 1000,
        "B" * 800,
        "C" * 400,
    ]

    result = evaluate_chunks(
        chunks,
        chunk_size=1000,
        short_threshold=500,
    )

    assert result["chunk_count"] == 3
    assert result["avg_length"] == pytest.approx(733.3333)
    assert result["min_length"] == 400
    assert result["max_length"] == 1000
    assert result["short_chunk_count"] == 1
    assert result["utilization"] == pytest.approx(0.733333)


def test_evaluate_chunks_empty():
    result = evaluate_chunks([])

    assert result == {
        "chunk_count": 0,
        "avg_length": 0.0,
        "min_length": 0,
        "max_length": 0,
        "short_chunk_count": 0,
        "utilization": 0.0,
    }


@pytest.mark.parametrize(
    "chunk_size,short_threshold",
    [
        (0, 500),
        (-1, 500),
        (1000, -1),
    ],
)
def test_evaluate_chunks_invalid_parameters(chunk_size, short_threshold):
    with pytest.raises(ValueError):
        evaluate_chunks(
            ["Example"],
            chunk_size=chunk_size,
            short_threshold=short_threshold,
        )


def test_evaluate_threshold_logic():
    chunks = [
        "A" * 499,
        "B" * 500,
        "C" * 501,
    ]

    result = evaluate_chunks(
        chunks,
        chunk_size=1000,
        short_threshold=500,
    )

    assert result["short_chunk_count"] == 1
