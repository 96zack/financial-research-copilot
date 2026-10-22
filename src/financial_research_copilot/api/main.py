from fastapi import FastAPI

from financial_research_copilot.api.schemas import (
    AskRequest,
    AskResponse,
)

app = FastAPI(
    title="Financial Research Copilot",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    return AskResponse(
        answer=f"Mock answer for: {request.question}",
        sources=[],
    )
