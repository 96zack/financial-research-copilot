from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

from financial_research_copilot.ingestion.document_pipeline import (
    ingest_and_chunk_filing,
)
from financial_research_copilot.retrieval.search import top_k_search
from financial_research_copilot.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

store = VectorStore(str(PROJECT_ROOT / "data" / "vector_store" / "apple_10k"))
model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

start = perf_counter()
if (Path(store.directory) / "manifest.json").exists():
    print("Loading cached embeddings...")
    document_embeddings, chunks = store.load(model_name)
else:
    print("Generating document embeddings...")

    result = ingest_and_chunk_filing(
        cik="0000320193",
        form_type="10-K",
        user_agent="FinancialResearchCopilot zackchen96@example.com",
    )

    # Extract chunks using your existing pipeline structure.
    chunks = result["chunks"]

    document_embeddings = model.encode(
        [chunk.text for chunk in chunks],
        show_progress_bar=True,
    )

    store.save(document_embeddings, chunks, model_name)

elapsed = perf_counter() - start
print(f"Document preparation time: {elapsed:.3f}s")

question = (
    "What risks does Apple identify related to artificial intelligence technologies?"
)

question_embedding = model.encode(
    question,
    show_progress_bar=True,
)

retrieveal_result = top_k_search(
    query_vector=question_embedding.tolist(),
    document_vectors=document_embeddings.tolist(),
    k=3,
)

for rank, (index, score) in enumerate(retrieveal_result, start=1):
    selected_chunk = chunks[index]

    print(f"Rank {rank}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Chunk ID: {selected_chunk.chunk_id}")
    print(f"First 200 Text: {selected_chunk.text[:200]}")
    print(f"URL: {selected_chunk.metadata.source_url}")
