
# Financial Research Copilot

A Python-based financial research assistant designed to help users
analyze SEC filings and answer questions about public companies.

> Status: Early development. The current application provides a
> FastAPI foundation with a mock question-answering endpoint.

## Project Goals

The planned application will:

- Retrieve and process SEC 10-K and 10-Q filings.
- Split financial documents into searchable chunks.
- Generate embeddings and retrieve relevant passages.
- Answer financial questions using an LLM.
- Provide citations to source documents.
- Evaluate retrieval and answer quality.

## Tech Stack

- Python 3.11
- FastAPI and Pydantic
- uv for dependency management
- pytest for automated testing
- Ruff for linting and formatting
- GitHub Actions for continuous integration

## Getting Started

### Prerequisites

Install Python 3.11 and uv.

### Installation

Clone the repository:

```bash
git clone https://github.com/96zack/financial-research-copilot.git
cd financial-research-copilot
```

Install dependencies:

```bash
uv sync --frozen
```

### Run the API

```bash
uv run uvicorn financial_research_copilot.api.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API documentation.

## API Endpoints

### GET /health

Health check endpoint.

Example response:

```json
{
  "status": "ok"
}
```

### POST /ask

Accepts a financial question and returns a mock answer.

Example request:

```json
{
  "question": "What was Apple's revenue?"
}
```

Example response:

```json
{
  "answer": "Mock answer for: What was Apple's revenue?",
  "sources": []
}
```

The endpoint currently returns mock responses. Document retrieval
and LLM integration are planned for later development.

## Development

Run linting:

```bash
uv run ruff check .
```

Check formatting:

```bash
uv run ruff format --check .
```

Run tests:

```bash
uv run pytest -v
```

## Continuous Integration

GitHub Actions automatically runs linting, formatting checks,
and automated tests on pushes and pull requests.

## Roadmap

- [x] FastAPI application setup
- [x] Pydantic request and response schemas
- [x] Automated API tests
- [x] Ruff linting and formatting
- [x] GitHub Actions CI
- [ ] SEC filing ingestion
- [ ] Document chunking
- [ ] Embeddings and vector search
- [ ] Retrieval-augmented generation
- [ ] Evaluation framework
- [ ] Deployment

## Disclaimer

This project is for educational and research purposes.
It does not provide investment advice.

