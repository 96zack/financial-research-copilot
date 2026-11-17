from financial_research_copilot.models.chunk import DocumentChunk


def deduplicate_chunks(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """
    Remove exact-text duplicates while preserving
    the original order and first occurrence.
    """
    seen = set()
    unique_chunks = []

    for chunk in chunks:
        # TODO 1: Check whether chunk.text has been seen.
        if chunk.text in seen:
            continue

        # TODO 2: If yes, skip this chunk.

        # TODO 3: Otherwise, record the text
        # and append the chunk to unique_chunks.
        seen.add(chunk.text)
        unique_chunks.append(chunk)

    return unique_chunks
