import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def generate_answer(question: str, context: str) -> str:
    # TODO 1: Validate question and context
    if not question.strip():
        raise ValueError("question cannot be empty")

    if not context.strip():
        raise ValueError("context cannot be empty")

    # TODO 2: Read OPENAI_API_KEY
    # Raise ValueError if missing
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    # TODO 3: Read OPENAI_MODEL
    # Default to "gpt-6-luna"
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")

    # TODO 4: Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # TODO 5: Call client.responses.create()
    # Pass a grounding instruction and the question/context
    response = client.responses.create(
        model=model,
        instructions=(
            "Answer using only the provided SEC filing excerpts. "
            "If the evidence is insufficient, say so."
        ),
        input=f"Context:\n{context}\n\nQuestion:\n{question}",
    )

    # TODO 6: Return response.output_text
    return response.output_text
