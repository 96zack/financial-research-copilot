from financial_research_copilot.ingestion.pipeline import (
    ingest_latest_filing,
)
from financial_research_copilot.processing.chunk_builder import (
    build_document_chunks,
)
from financial_research_copilot.processing.chunk_evaluator import (
    evaluate_chunks,
)
from financial_research_copilot.processing.chunker import (
    split_text_by_paragraph,
)
from financial_research_copilot.processing.html_parser import (
    remove_page_footers,
)


def ingest_and_chunk_filing(
    cik: str,
    form_type: str,
    user_agent: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    # TODO 1: Call ingest_latest_filing()
    filing = ingest_latest_filing(
        cik=cik,
        form_type=form_type,
        user_agent=user_agent,
    )
    # TODO 2: Remove page footers from filing["text"]
    cleaned_text = remove_page_footers(filing["text"])
    # TODO 3: Split the cleaned text into paragraph-aware chunks
    text_chunks = split_text_by_paragraph(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # TODO 4: Convert the strings into DocumentChunk objects
    doc_chunks = build_document_chunks(filing, text_chunks)
    # TODO 5: Evaluate the text chunks
    chunk_eva = evaluate_chunks(text_chunks, chunk_size)
    # TODO 6: Return filing, structured chunks, and metrics
    return {
        "filing": filing,
        "chunks": doc_chunks,
        "metrics": chunk_eva,
    }
