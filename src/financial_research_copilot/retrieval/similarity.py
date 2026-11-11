import math


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    # TODO 1: Validate equal vector lengths
    if len(vector_a) != len(vector_b):
        raise ValueError("vector lengths must be equal")
    # TODO 2: Reject empty vectors
    if not vector_a or not vector_b:
        raise ValueError("Input Vector can not be empty")
    # TODO 3: Compute the dot product
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    # TODO 4: Compute the Euclidean norm of each vector
    norm_vec_a = math.sqrt(sum(x**2 for x in vector_a))
    norm_vec_b = math.sqrt(sum(x**2 for x in vector_b))
    # TODO 5: Reject zero-magnitude vectors
    if norm_vec_a == 0 or norm_vec_b == 0:
        raise ValueError("Zero Magnitude Vectors")
    # TODO 6: Return cosine similarity
    return dot_product / (norm_vec_a * norm_vec_b)
