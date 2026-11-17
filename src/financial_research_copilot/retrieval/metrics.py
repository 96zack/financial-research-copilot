def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k should be greater than 0")

    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError("duplicate ids detected")

    top_k_ids = set(retrieved_ids[:k])
    hits = top_k_ids & relevant_ids

    return len(hits) / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k should be greater than 0")

    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError("duplicate ids detected")

    if len(relevant_ids) == 0:
        return 0

    top_k_ids = set(retrieved_ids[:k])
    hits = top_k_ids & relevant_ids

    return len(hits) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k should be greater than 0")

    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError("duplicate ids detected")

    for index, chunk_id in enumerate(retrieved_ids[:k]):
        if chunk_id in relevant_ids:
            return 1 / (index + 1)

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:
    if not reciprocal_ranks:
        return 0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
