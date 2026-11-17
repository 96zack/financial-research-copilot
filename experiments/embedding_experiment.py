from sentence_transformers import SentenceTransformer

from financial_research_copilot.retrieval.similarity import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

sentences = [
    "Apple relies on a limited number of component suppliers.",
    "Supply chain disruptions could affect Apple's operations.",
    "Apple reported strong revenue growth.",
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)

similarity_12 = cosine_similarity(
    embeddings[0].tolist(),
    embeddings[1].tolist(),
)

print(f"Similarity 1-2: {similarity_12:.4f}")

similarity_13 = cosine_similarity(
    embeddings[0].tolist(),
    embeddings[2].tolist(),
)

print(f"Similarity 1-3: {similarity_13:.4f}")

similarity_23 = cosine_similarity(
    embeddings[1].tolist(),
    embeddings[2].tolist(),
)

print(f"Similarity 2-3: {similarity_23:.4f}")
