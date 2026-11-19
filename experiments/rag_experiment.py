from pathlib import Path

from sentence_transformers import SentenceTransformer

from financial_research_copilot.generation.rag_pipeline import (
    answer_question_with_citations,
)
from financial_research_copilot.retrieval.search import top_k_search
from financial_research_copilot.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

store = VectorStore(str(PROJECT_ROOT / "data" / "vector_store" / "apple_10k"))
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Load your existing VectorStore
if (Path(store.directory) / "manifest.json").exists():
    print("Loading cached embeddings...")
    document_embeddings, chunks = store.load(model_name)

# 2. Load the same sentence-transformer model used for indexing
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

question = "What factors could affect Apple's gross margin?"

# 3. Encode the user's question
question_embedding = model.encode(
    question,
    show_progress_bar=True,
)

# 4. Run top_k_search() against the cached embeddings
retrieveal_result = top_k_search(
    query_vector=question_embedding.tolist(),
    document_vectors=document_embeddings.tolist(),
    k=3,
)

# optional
for rank, (index, score) in enumerate(retrieveal_result, start=1):
    print(f"\n[Source {rank}] Similarity: {score:.4f}")
    print(chunks[index].text[:400])

# 5. Extract the corresponding DocumentChunk.text values
# retrieved_chunks = [chunks[index].text for index, score in retrieveal_result]
retrieved_chunks = [chunks[index] for index, score in retrieveal_result]

# 6. Pass those passages to answer_question()
# answer = answer_question(question, retrieved_chunks)
response = answer_question_with_citations(
    question,
    retrieved_chunks,
)

# 7. Print the retrieved passages and generated answer
print("\n=== Generated Answer ===")
print(response.answer)

print("\n=== Citation Validation ===")
print(f"Valid citations: {response.citations_valid}")

print("\n=== Supporting Sources ===")
for source in response.sources:
    print(f"\n[Source {source.source_number}]")
    print(f"Chunk ID: {source.chunk_id}")
    print(f"Text: {source.text[:500]}")
