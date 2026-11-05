import hashlib

from financial_research_copilot.models.chunk import (
    ChunkMetadata,
    DocumentChunk,
)


def build_document_chunks(
    filing: dict,
    chunks: list[str],
) -> list[DocumentChunk]:
    source_url = filing["url"]

    document_id = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]

    results = []

    for index, text in enumerate(chunks):
        # TODO 1: Construct ChunkMetadata
        # Map filing["url"] to source_url
        # Use index for chunk_index
        metadata = ChunkMetadata(
            company_name=filing["company_name"],
            cik=filing["cik"],
            form=filing["form"],
            filing_date=filing["filing_date"],
            source_url=source_url,
            chunk_index=index,
        )

        # TODO 2: Construct DocumentChunk
        chunk_id = f"{document_id}-{index:04d}"
        doc_chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
        )

        # TODO 3: Append DocumentChunk to results
        results.append(doc_chunk)

    return results
