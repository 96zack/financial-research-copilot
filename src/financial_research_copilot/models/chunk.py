from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    # Required fields:
    company_name: str
    cik: str
    form: str
    filing_date: str
    source_url: str
    chunk_index: int = Field(ge=0)


class DocumentChunk(BaseModel):
    # Required fields:
    chunk_id: str
    text: str = Field(min_length=1)
    metadata: ChunkMetadata
