from pathlib import Path

from sentence_transformers import SentenceTransformer

from financial_research_copilot.evaluation.citation_metrics import (
    citation_coverage,
    citation_precision,
)
from financial_research_copilot.evaluation.grounding import (
    abstention_correct,
    grounding_score,
)
from financial_research_copilot.evaluation.grounding_judge import evaluate_grounding
from financial_research_copilot.generation.rag_pipeline import (
    answer_question_structured,
    format_context,
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

# question = "What factors could affect Apple's gross margin?"
question = "What exact gross margin will Apple report in fiscal 2028?"

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
# for rank, (index, score) in enumerate(retrieveal_result, start=1):
#     print(f"\n[Source {rank}] Similarity: {score:.4f}")
#     print(chunks[index].text[:400])

# 5. Extract the corresponding DocumentChunk.text values
# retrieved_chunks = [chunks[index].text for index, score in retrieveal_result]
retrieved_chunks = [chunks[index] for index, score in retrieveal_result]

# 6. Pass those passages to answer_question()
# answer = answer_question(question, retrieved_chunks)
rag_response = answer_question_structured(
    question,
    retrieved_chunks,
)

answer = rag_response.result.answer
sources = rag_response.sources

# 1. Citation coverage
coverage = citation_coverage(answer)

# 2. Citation precision
available_source_numbers = {source.source_number for source in sources}
precision = citation_precision(answer, available_source_numbers)

# 3. Grounding evaluation
context = format_context([source.text for source in sources])

grounding_evaluation = evaluate_grounding(
    answer=answer,
    context=context,
)

score = grounding_score(grounding_evaluation)

print("\n=== RAG Quality Evaluation ===")
print(f"Citation coverage: {coverage:.2%}")
print(f"Citation precision: {precision:.2%}")
print(f"Grounding score: {score:.2%}")
print(f"Citations valid: {rag_response.citations_valid}")
print(f"Insufficient evidence: {rag_response.result.insufficient_evidence}")

print("\n=== Claim-Level Evaluation ===")

for index, claim in enumerate(grounding_evaluation.claims, start=1):
    print(f"\nClaim {index}: {claim.claim}")
    print(f"Sources: {claim.cited_sources}")
    print(f"Supported: {claim.supported}")
    print(f"Reason: {claim.explanation}")

expected_abstention = True

abstention_passed = abstention_correct(
    predicted_insufficient_evidence=rag_response.result.insufficient_evidence,
    expected_abstention=expected_abstention,
)

print(f"Abstention correct: {abstention_passed}")
