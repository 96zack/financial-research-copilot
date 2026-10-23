
from fastapi.testclient import TestClient

from financial_research_copilot.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_valid_question():
    response = client.post(
        "/ask",
        json={"question": "What was Apple's revenue?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == (
        "Mock answer for: What was Apple's revenue?"
    )
    assert data["sources"] == []


def test_ask_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422
