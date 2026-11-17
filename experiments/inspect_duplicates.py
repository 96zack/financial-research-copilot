from collections import defaultdict
from pathlib import Path

from financial_research_copilot.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

store = VectorStore(
    str(PROJECT_ROOT / "data" / "vector_store" / "apple_10k")
)

embeddings, chunks = store.load(MODEL_NAME)

# TODO 1:
# Create a dictionary mapping each chunk's text
# to all chunk IDs containing that exact text.
#
# Hint: defaultdict(list)
text_to_ids = defaultdict(list)

for chunk in chunks:
    # Your code here
    text_to_ids[chunk.text].append(chunk.chunk_id)

# TODO 2:
# Keep only groups containing more than one chunk ID.

duplicates = {
    text: ids
    for text, ids in text_to_ids.items()
    if len(ids) > 1
}

# TODO 3:
# Print:
# - Total number of chunks
# - Number of duplicated text groups
# - Number of redundant chunks
# - First 5 duplicate groups and their chunk IDs
total_chunks = len(chunks)
duplicate_groups = len(duplicates)

redundant_chunks = sum(
    len(ids) - 1
    for ids in duplicates.values()
)

print(f"Total chunks: {total_chunks}")
print(f"Duplicate groups: {duplicate_groups}")
print(f"Redundant chunks: {redundant_chunks}")
print(f"Redundancy rate: {redundant_chunks / total_chunks:.2%}")

print("\nFirst 5 duplicate groups:")

for text, ids in list(duplicates.items())[:5]:
    print(f"\nChunk IDs: {ids}")
    print(f"Text preview: {text[:150]}...")
