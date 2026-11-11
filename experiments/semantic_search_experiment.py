from sentence_transformers import SentenceTransformer

from financial_research_copilot.ingestion.document_pipeline import (
    ingest_and_chunk_filing,
)
from financial_research_copilot.retrieval.search import top_k_search

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

result = ingest_and_chunk_filing(
    cik="0000320193",
    form_type="10-K",
    user_agent="FinancialResearchCopilot zackchen96@example.com",
)

chunk_texts = [chunk.text for chunk in result["chunks"]]

document_embeddings = model.encode(
    chunk_texts,
    show_progress_bar=True,
)

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

# print(retrieveal_result)

i = 1
for r in retrieveal_result:
    print("Rank " + str(i))
    index = r[0]
    score = r[1]
    print(f"Similarity Score {score:.4f}")
    selected_chunk = result["chunks"][index]
    print(f"Chunk ID :{selected_chunk.chunk_id}")
    print(f"First 200 Text :{selected_chunk.text[:200]}")
    print(f"URL :{selected_chunk.metadata.source_url}")
    i += 1
