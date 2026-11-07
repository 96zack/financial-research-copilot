from statistics import mean


def evaluate_chunks(
    chunks: list[str],
    chunk_size: int = 1000,
    short_threshold: int = 500,
) -> dict:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if short_threshold < 0:
        raise ValueError("short_threshold must be non-negative")

    # TODO 1: Handle an empty list safely.
    # Decide what to return for zero chunks.
    if not chunks:
        return {
            "chunk_count": 0,
            "avg_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "short_chunk_count": 0,
            "utilization": 0.0,
        }

    # TODO 1: Create a list of chunk lengths
    lengths = [len(chunk) for chunk in chunks]

    # TODO 2: Calculate average length
    avg_length = mean(lengths)

    # TODO 3: Count chunks shorter than short_threshold
    short_chunk_count = sum(length < short_threshold for length in lengths)

    # TODO 4: Return the metrics
    return {
        "chunk_count": len(chunks),
        "avg_length": avg_length,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "short_chunk_count": short_chunk_count,
        "utilization": avg_length / chunk_size,
    }
