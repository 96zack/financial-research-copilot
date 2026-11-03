def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    if not text:
        return []

    chunks = []
    start = 0

    # TODO:
    # 1. Calculate the end position
    # 2. Extract text[start:end]
    # 3. Append it to chunks
    # 4. Advance start, accounting for overlap
    # 5. Stop when the final chunk reaches the end of text
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        step = chunk_size - chunk_overlap
        start += step

    return chunks