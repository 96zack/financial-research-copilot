from financial_research_copilot.generation.llm_client import generate_answer


def format_context(chunks: list[str]) -> str:
    """Format retrieved passages into numbered evidence blocks."""
    formatted_chunks = []
    for ind, text in enumerate(chunks, start=1):
        formatted_chunks.append(f"[Source {ind}]\n{text}")
    return "\n\n".join(formatted_chunks)


def answer_question(
    question: str,
    retrieved_chunks: list[str],
) -> str:
    """Generate an answer using retrieved SEC filing passages."""

    # TODO 1: Validate retrieved_chunks is not empty
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")

    # TODO 2: Call format_context(retrieved_chunks)
    formatted_context = format_context(retrieved_chunks)

    # TODO 3: Call generate_answer(question, context)
    answer = generate_answer(question, formatted_context)

    # TODO 4: Return the generated answer
    return answer
