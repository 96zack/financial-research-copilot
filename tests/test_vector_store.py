import numpy as np
import pytest

from financial_research_copilot.models.chunk import (
    ChunkMetadata,
    DocumentChunk,
)
from financial_research_copilot.retrieval.vector_store import VectorStore


def test_vector_store_round_trip(tmp_path):
    # 1. Create an isolated temporary directory
    store = VectorStore(str(tmp_path))

    # 2. Create two fake embeddings
    embeddings = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    # 3. Create sample metadata
    metadata = ChunkMetadata(
        company_name="Apple Inc.",
        cik="0000320193",
        form="10-K",
        filing_date="2026-09-26",
        source_url="https://example.com/apple-10k",
        chunk_index=1
    )

    # 4. Create two sample chunks
    chunks = [
        DocumentChunk(
            chunk_id="test-chunk-001",
            text="Apple faces supply chain risks.",
            metadata=metadata,
        ),
        DocumentChunk(
            chunk_id="test-chunk-002",
            text="Apple reported revenue growth.",
            metadata=metadata,
        ),
    ]

    # 5. Save
    model_name = "test-embedding-model"
    store.save(embeddings, chunks, model_name)

    # 6. Load
    loaded_embeddings, loaded_chunks = store.load(model_name)

    # 7. Compare the original and loaded data
    np.testing.assert_array_equal(loaded_embeddings, embeddings)

    assert loaded_chunks == chunks

    # 8. Verify that all three files were created
    assert (tmp_path / "embeddings.npz").exists()
    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "manifest.json").exists()


def test_model_mismatch(tmp_path):
    store = VectorStore(str(tmp_path))
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    metadata = ChunkMetadata(
        company_name="Apple Inc.",
        cik="0000320193",
        form="10-K",
        filing_date="2026-09-26",
        source_url="https://example.com/apple-10k",
        chunk_index=1
    )

    # Reuse one valid DocumentChunk from your round-trip test.
    chunks = [
        DocumentChunk(
            chunk_id="test-chunk-001",
            text="Apple faces supply chain risks.",
            metadata=metadata,
        ),
    ]

    store.save(embeddings, chunks, "model-A")

    with pytest.raises(ValueError, match="Model not matching"):
        store.load("model-B")


def test_count_mismatch(tmp_path):
    store = VectorStore(str(tmp_path))
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    metadata = ChunkMetadata(
        company_name="Apple Inc.",
        cik="0000320193",
        form="10-K",
        filing_date="2026-09-26",
        source_url="https://example.com/apple-10k",
        chunk_index=1
    )

    # One valid chunk, but two embeddings.
    chunks = [
        DocumentChunk(
            chunk_id="test-chunk-001",
            text="Apple faces supply chain risks.",
            metadata=metadata,
        ),
    ]

    with pytest.raises(ValueError):
        store.save(embeddings, chunks, "test-model")


def test_invalid_dimensions(tmp_path):
    store = VectorStore(str(tmp_path))
    embeddings = np.array([1.0, 0.0])

    with pytest.raises(ValueError):
        store.save(embeddings, [], "test-model")


def test_missing_cache(tmp_path):
    store = VectorStore(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        store.load("test-model")
