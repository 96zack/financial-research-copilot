import json
from pathlib import Path

from financial_research_copilot.evaluation.citation_metrics import (
    citation_coverage,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    directory = project_root / "data" / "evaluation"

    input_path = directory / "rag_eval_baseline.json"
    output_path = directory / "rag_eval_baseline_v2.json"

    with input_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    old_coverage = report["summary"]["citation_coverage"]

    for result in report["results"]:
        previous = result["citation_coverage"]
        updated = citation_coverage(result["answer"])

        result["citation_coverage"] = updated

        print(f"{result['case_id']}: {previous:.2%} -> {updated:.2%}")

    new_coverage = sum(
        result["citation_coverage"] for result in report["results"]
    ) / len(report["results"])

    report["summary"]["citation_coverage"] = new_coverage
    report["configuration"]["citation_metric_version"] = "v2"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nOverall: {old_coverage:.2%} -> {new_coverage:.2%}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
