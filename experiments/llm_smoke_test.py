from financial_research_copilot.generation.llm_client import generate_answer

question = "What factors could affect Apple's gross margins?"

context = (
    "Apple's gross margins may be affected by changes in "
    "product mix, component costs, and competitive pricing."
)

answer = generate_answer(question, context)

print(answer)
