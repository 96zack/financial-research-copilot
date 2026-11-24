from financial_research_copilot.generation.llm_client import (
    generate_answer,
)


def main() -> None:
    question = "Your existing question"
    context = "Your existing context"

    answer = generate_answer(question, context)
    print(answer)


if __name__ == "__main__":
    main()
