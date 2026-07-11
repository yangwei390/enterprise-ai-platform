import argparse
import time
from pathlib import Path
from typing import Any

from backend.app.chat import ChatRequest, ChatService

from evaluation.evaluator import EvaluationQuestion, Evaluator
from evaluation.report import build_report, print_summary, write_report
from evaluation.v2.runner import EvaluationRunnerV2

DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "questions.yaml"
DEFAULT_REPORT_PATH = Path(__file__).parent / "report.json"


def load_questions(path: Path = DEFAULT_DATASET_PATH) -> list[EvaluationQuestion]:
    raw_items = _load_simple_yaml_list(path)
    return [
        EvaluationQuestion(
            id=str(item["id"]),
            question=str(item["question"]),
            expected_documents=[str(value) for value in item.get("expected_documents", [])],
            expected_chunks=[int(value) for value in item.get("expected_chunks", [])],
            expected_keywords=[str(value) for value in item.get("expected_keywords", [])],
            knowledge_base_id=_optional_int(item.get("knowledge_base_id")),
            top_k=int(item.get("top_k", 5)),
            score_threshold=_optional_float(item.get("score_threshold")),
        )
        for item in raw_items
    ]


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    chat_service: ChatService | None = None,
) -> dict[str, Any]:
    questions = load_questions(dataset_path)
    service = chat_service or ChatService()
    evaluator = Evaluator()
    results = []

    for question in questions:
        started_at = time.perf_counter()
        response = service.chat(
            ChatRequest(
                query=question.question,
                knowledge_base_id=question.knowledge_base_id,
                top_k=question.top_k,
                score_threshold=question.score_threshold,
                enable_memory=False,
                enable_tools=False,
            )
        )
        total_ms = round((time.perf_counter() - started_at) * 1000, 2)
        latency = _extract_latency(response.metadata, total_ms)
        results.append(evaluator.evaluate(question, response, latency))

    report = build_report(results)
    write_report(report, report_path)
    print_summary(report)
    return report


def _extract_latency(metadata: dict, total_ms: float) -> dict[str, float]:
    return {
        "total_ms": total_ms,
        "retriever_ms": _float(metadata.get("retriever_latency_ms")),
        "llm_ms": _float(metadata.get("llm_latency_ms")),
    }


def _load_simple_yaml_list(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current_item: dict[str, Any] | None = None
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and not raw_line.startswith(" "):
            if current_item is not None:
                items.append(current_item)
            current_item = {}
            current_list_key = None
            key, value = _split_key_value(stripped[2:])
            current_item[key] = _parse_scalar(value)
            continue

        if current_item is None:
            continue

        if stripped.startswith("- ") and current_list_key:
            current_item.setdefault(current_list_key, []).append(
                _parse_scalar(stripped[2:])
            )
            continue

        key, value = _split_key_value(stripped)
        if value == "":
            current_item[key] = []
            current_list_key = key
        else:
            current_item[key] = _parse_scalar(value)
            current_list_key = None

    if current_item is not None:
        items.append(current_item)
    return items


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        return text, ""
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> str | int | float | bool | None:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _float(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise AI Platform evaluation")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument("--target", default=None)
    parser.add_argument("--tags", action="append", default=[])
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--save-baseline", default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if not args.suite:
        report = run_evaluation()
        return 0 if int(report.get("fail", 0)) == 0 else 1

    suite = _resolve_suite_arg(args.suite)
    tags = [*args.tags]
    if args.target:
        tags.append(args.target)
    report_v2 = EvaluationRunnerV2(
        concurrency=args.concurrency,
        fail_fast=args.fail_fast,
    ).run_suite(
        suite,
        case_ids=args.case_ids,
        tags=tags,
        baseline=args.baseline,
        compare=args.compare,
        save_baseline=args.save_baseline,
        output=args.output,
    )
    _print_v2_summary(report_v2.model_dump(mode="json"))
    return 0 if report_v2.passed else 1


def _resolve_suite_arg(suite: str) -> str:
    if "/" in suite or suite.endswith((".yaml", ".yml", ".json")):
        return suite
    return str(Path("evaluation/v2/fixtures/suites") / f"{suite}.yaml")


def _print_v2_summary(report: dict[str, Any]) -> None:
    print("\nEvaluation Summary V2")
    print("=====================")
    print(f"PASS {report.get('passed_cases', 0)}")
    print(f"FAIL {report.get('failed_cases', 0)}")
    print(f"SKIP {report.get('skipped_cases', 0)}")
    print(f"Pass Rate {round(float(report.get('pass_rate', 0)) * 100, 2)}%")
    print(f"Duration {report.get('duration_ms', 0)}ms")


if __name__ == "__main__":
    raise SystemExit(main())
