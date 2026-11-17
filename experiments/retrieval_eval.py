from pathlib import Path

from sentence_transformers import SentenceTransformer

from financial_research_copilot.retrieval.search import top_k_search
from financial_research_copilot.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUESTIONS = {
    "supplier": (
        "What risks does Apple face from relying on "
        "a limited number of suppliers?"
    ),
    "margin": (
        "What factors could negatively affect Apple's gross margins?"
    ),
    "ai": (
        "What risks does Apple identify related to "
        "artificial intelligence technologies?"
    ),
}


def main():
    store = VectorStore(
        str(PROJECT_ROOT / "data" / "vector_store" / "apple_10k")
    )

    embeddings, chunks = store.load(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    for query_name, question in QUESTIONS.items():
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query_name}")
        print(f"QUESTION: {question}")

        query_vector = model.encode(question)

        results = top_k_search(
            query_vector=query_vector.tolist(),
            document_vectors=embeddings.tolist(),
            k=10,
        )

        for rank, (index, score) in enumerate(results, start=1):
            chunk = chunks[index]

            print(f"\nRank {rank} | Score: {score:.4f}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Text:\n{chunk.text[:1200]}")


if __name__ == "__main__":
    main()
