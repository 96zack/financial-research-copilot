import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer

from experiments.rag_eval_dataset import EVAL_CASES, RAGEvalCase
from financial_research_copilot.evaluation.citation_metrics import (
    citation_coverage,
    citation_precision,
)
from financial_research_copilot.evaluation.grounding import (
    abstention_correct,
    grounding_score,
)
from financial_research_copilot.evaluation.grounding_judge import (
    evaluate_grounding,
)
from financial_research_copilot.generation.rag_pipeline import (
    answer_question_structured,
    format_context,
)
from financial_research_copilot.models.chunk import DocumentChunk
from financial_research_copilot.retrieval.search import top_k_search
from financial_research_copilot.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RAGEvalResult:
    case_id: str
    question: str
    answer: str

    citation_coverage: float
    citation_precision: float
    grounding_score: float

    citations_valid: bool
    insufficient_evidence: bool
    abstention_correct: bool


def summarize_results(results: list[RAGEvalResult]) -> dict[str, float]:
    """Compute aggregate metrics across evaluation cases."""

    if not results:
        raise ValueError("results cannot be empty")

    cit_coverage, cit_precision, ground_score, abstention, validity = 0, 0, 0, 0, 0
    for r in results:
        # TODO 1:
        # Calculate average citation coverage.
        cit_coverage = cit_coverage + r.citation_coverage

        # TODO 2:
        # Calculate average citation precision.
        cit_precision = cit_precision + r.citation_precision

        # TODO 3:
        # Calculate average grounding score.
        ground_score = ground_score + r.grounding_score

        # TODO 4:
        # Calculate abstention accuracy.
        abstention = abstention + r.abstention_correct

        # TODO 5:
        # Calculate citation validity rate.
        validity = validity + r.citations_valid

    return {
        "citation_coverage": cit_coverage / len(results),
        "citation_precision": cit_precision / len(results),
        "grounding_score": ground_score / len(results),
        "abstention_accuracy": abstention / len(results),
        "citation_validity_rate": validity / len(results),
    }


def evaluate_case(
    case: RAGEvalCase,
    retrieved_chunks: list[DocumentChunk],
) -> RAGEvalResult:
    """Evaluate a single RAG evaluation case."""

    rag_response = answer_question_structured(
        question=case.question,
        retrieved_chunks=retrieved_chunks,
    )

    answer = rag_response.result.answer
    sources = rag_response.sources

    # TODO 1: Calculate citation coverage.
    coverage = citation_coverage(answer)

    # TODO 2: Calculate citation precision using
    # available source numbers.
    available_source_numbers = {source.source_number for source in sources}
    precision = citation_precision(answer, available_source_numbers)

    # TODO 3: Reconstruct the formatted context,
    # call evaluate_grounding(), and calculate grounding score.
    context = format_context([source.text for source in sources])

    grounding_evaluation = evaluate_grounding(
        answer=answer,
        context=context,
    )
    score = grounding_score(grounding_evaluation)

    # TODO 4: Check abstention correctness using
    # case.should_abstain.
    abstention_passed = abstention_correct(
        predicted_insufficient_evidence=rag_response.result.insufficient_evidence,
        expected_abstention=case.should_abstain,
    )

    # TODO 5: Construct and return RAGEvalResult.
    return RAGEvalResult(
        case_id=case.case_id,
        question=case.question,
        answer=answer,
        citation_coverage=coverage,
        citation_precision=precision,
        grounding_score=score,
        citations_valid=rag_response.citations_valid,
        insufficient_evidence=rag_response.result.insufficient_evidence,
        abstention_correct=abstention_passed,
    )


def run_evaluation(
    cases: list[RAGEvalCase],
    k: int = 3,
) -> list[RAGEvalResult]:
    """Run RAG evaluation across multiple test cases."""

    project_root = Path(__file__).resolve().parents[1]

    store = VectorStore(str(project_root / "data" / "vector_store" / "apple_10k"))

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # 1. Load vector store once
    if not (Path(store.directory) / "manifest.json").exists():
        raise FileNotFoundError("Cached vector store not found")

    print("Loading cached embeddings...")
    document_embeddings, chunks = store.load(model_name)

    document_vectors = document_embeddings.tolist()

    # 2. Initialize embedding model once
    model = SentenceTransformer(
        model_name,
        device="cpu",
    )

    results = []

    for case in cases:
        print(f"Evaluating: {case.case_id}")

        # 3. Embed the question
        question_embedding = model.encode(
            case.question,
            show_progress_bar=False,
        )

        # 4. Retrieve top-k document chunks
        retrieval_result = top_k_search(
            query_vector=question_embedding.tolist(),
            document_vectors=document_vectors,
            k=k,
        )

        retrieved_chunks = [chunks[index] for index, _ in retrieval_result]

        # 5. Evaluate the case
        evaluation = evaluate_case(case, retrieved_chunks)

        # 6. Append the result
        results.append(evaluation)

    return results


def save_evaluation_report(
    results: list[RAGEvalResult],
    output_path: Path,
    k: int,
) -> None:
    """Save evaluation results and configuration as JSON."""

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "generation_model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "grounding_model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "top_k": k,
            "num_cases": len(results),
        },
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def main() -> None:
    k = 3

    results = run_evaluation(EVAL_CASES, k=k)
    summary = summarize_results(results)

    print("\n=== RAG Evaluation Summary ===")

    for metric, value in summary.items():
        print(f"{metric}: {value:.2%}")

    project_root = Path(__file__).resolve().parents[1]

    output_path = project_root / "data" / "evaluation" / "rag_eval_baseline.json"

    save_evaluation_report(results, output_path, k)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
