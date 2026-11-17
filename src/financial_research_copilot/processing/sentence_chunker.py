import re


def split_into_sentences(text: str) -> list[str]:
    """
    Split text at sentence boundaries while
    preserving punctuation.
    """
    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [sentence for sentence in sentences if sentence]


def split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    words = sentence.split()
    chunks = []
    current = ""

    for word in words:
        # Handle oversized words
        if len(word) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            while len(word) > chunk_size:
                chunks.append(word[:chunk_size])
                word = word[chunk_size:]

            if not word:
                continue

        separator = " " if current else ""
        candidate = current + separator + word

        if len(candidate) > chunk_size:
            if current:
                chunks.append(current)

            current = word
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def split_long_paragraph(
    text: str,
    chunk_size: int = 1000,
) -> list[str]:
    """
    Split a long paragraph into chunks,
    preferring sentence boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sentences = split_into_sentences(text)

    chunks = []
    current = ""

    for sentence in sentences:
        # TODO 3: Handle a sentence longer than chunk_size
        if len(sentence) > chunk_size:
            # Save any existing content first
            if current:
                chunks.append(current)
                current = ""

            # Split oversized sentence at word boundaries
            chunks.extend(split_long_sentence(sentence, chunk_size))
            continue

        # TODO 1 & 2: Accumulate sentences
        separator = " " if current else ""
        candidate = current + separator + sentence

        if len(candidate) > chunk_size:
            if current:
                chunks.append(current)

            current = sentence
        else:
            current = candidate

    # TODO 4: Save the final chunk
    if current:
        chunks.append(current)

    return chunks
