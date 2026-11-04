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


def split_text_by_paragraph(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            # Save current_chunk
            if current_chunk:
                chunks.append(current_chunk)
            # Split oversized paragraph
            chunks.extend(split_text(paragraph, chunk_size, chunk_overlap))
            # Reset current_chunk
            current_chunk = ""

        else:
            # Build candidate string
            candidate = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )
            # If it fits, update current_chunk
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            # Otherwise, save current_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # and start a new one
                current_chunk = paragraph

    # Append final current_chunk if nonempty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
