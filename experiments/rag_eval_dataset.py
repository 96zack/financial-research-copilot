from dataclasses import dataclass


@dataclass(frozen=True)
class RAGEvalCase:
    case_id: str
    question: str
    expected_source_topics: tuple[str, ...]
    should_abstain: bool


EVAL_CASES = [
    RAGEvalCase(
        case_id="gross_margin",
        question="What factors could affect Apple's gross margin?",
        expected_source_topics=("gross margin", "product mix"),
        should_abstain=False,
    ),
    RAGEvalCase(
        case_id="product_margin",
        question="What contributed to changes in Apple's product gross margin in 2026?",
        expected_source_topics=("product mix", "tariff refunds", "memory"),
        should_abstain=False,
    ),
    RAGEvalCase(
        case_id="services_margin",
        question="What factors affected Apple's Services gross margin in 2026?",
        expected_source_topics=("services gross margin",),
        should_abstain=False,
    ),
    RAGEvalCase(
        case_id="iphone_sales",
        question="What contributed to Apple's iPhone sales growth in 2026?",
        expected_source_topics=("iphone net sales", "pro models"),
        should_abstain=False,
    ),
    RAGEvalCase(
        case_id="unsupported_forecast",
        question="What exact gross margin will Apple report in fiscal 2028?",
        expected_source_topics=(),
        should_abstain=True,
    ),
]
