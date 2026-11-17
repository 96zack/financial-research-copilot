import json
from pathlib import Path

import numpy as np

from financial_research_copilot.models.chunk import DocumentChunk


class VectorStore:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def save(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
        model_name: str,
    ) -> None:
        # TODO 1: Validate embeddings is a 2D array
        if embeddings.ndim != 2:
            raise ValueError("embeddings should be a 2d array")

        # TODO 2: Validate embeddings.shape[0] == len(chunks)
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding count must match chunk count")

        # TODO 3: Create directory if necessary
        self.directory.mkdir(parents=True, exist_ok=True)

        # TODO 4: Save embeddings using np.savez_compressed()
        np.savez_compressed(
            self.directory / "embeddings.npz",
            embeddings=embeddings,
        )

        # TODO 5: Serialize chunks using Pydantic's model_dump()
        chunk_data = [chunk.model_dump(mode="json") for chunk in chunks]
        (self.directory / "chunks.json").write_text(
            json.dumps(chunk_data, indent=2),
            encoding="utf-8",
        )

        # TODO 6: Write manifest.json with:
        # model_name, chunk_count, embedding_dimension
        output_json = {
            "model_name": model_name,
            "chunk_count": len(chunks),
            "embedding_dimension": embeddings.shape[1],
        }

        (self.directory / "manifest.json").write_text(
            json.dumps(output_json, indent=2),
            encoding="utf-8",
        )

    def load(
        self,
        expected_model_name: str,
    ) -> tuple[np.ndarray, list[DocumentChunk]]:
        """
        Load cached embeddings and chunks.

        Raise ValueError if the cache is incompatible.
        Raise FileNotFoundError if required files are missing.
        """

        # TODO 1: Read manifest.json using json.loads()
        manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )

        # TODO 2: Validate manifest["model_name"]
        # against expected_model_name
        if manifest["model_name"] != expected_model_name:
            raise ValueError("Model not matching")

        # TODO 3: Load embeddings.npz using np.load()
        # Retrieve the array stored under "embeddings"
        with np.load(self.directory / "embeddings.npz", allow_pickle=False) as data:
            embeddings_load = data["embeddings"]

        # TODO 4: Read chunks.json using json.loads()
        chunk_data_load = json.loads(
            (self.directory / "chunks.json").read_text(encoding="utf-8")
        )

        # TODO 5: Reconstruct DocumentChunk objects
        # using DocumentChunk.model_validate()
        chunks_load = [DocumentChunk.model_validate(item) for item in chunk_data_load]

        # TODO 6: Validate:
        # - embeddings.ndim == 2
        # - embeddings.shape[0] == len(chunks)
        # - embeddings.shape[0] == manifest["chunk_count"]
        # - embeddings.shape[1] == manifest["embedding_dimension"]
        if embeddings_load.ndim != 2:
            raise ValueError("Cached embeddings must be a 2D array")

        if (
            embeddings_load.shape[0] != len(chunks_load)
            or embeddings_load.shape[0] != manifest["chunk_count"]
            or embeddings_load.shape[1] != manifest["embedding_dimension"]
        ):
            raise ValueError("Cached embeddings do not match metadata")

        # TODO 7: Return (embeddings, chunks)
        return embeddings_load, chunks_load
