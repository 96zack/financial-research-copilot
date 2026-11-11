from financial_research_copilot.retrieval.similarity import cosine_similarity


def top_k_search(
    query_vector: list[float],
    document_vectors: list[list[float]],
    k: int = 3,
) -> list[tuple[int, float]]:
    """
    Return the top-k document indices and cosine similarity scores.

    Results must be ordered by descending similarity.
    """
    # TODO 1: Validate k > 0
    if k <= 0:
        raise ValueError("K must be greater than 0")

    # TODO 2: Return [] if document_vectors is empty
    if not document_vectors:
        return []

    # TODO 3: Calculate similarity between query and each document
    results = []
    for index, vec in enumerate(document_vectors):
        score = cosine_similarity(query_vector, vec)
        results.append((index, score))

    # TODO 4: Store (document_index, similarity_score) tuples

    # TODO 5: Sort by similarity, highest first
    sorted_results = sorted(results, key=lambda item: item[1], reverse=True)

    # TODO 6: Return at most k results
    return sorted_results[:k]
