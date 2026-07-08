import json
from pathlib import Path
from typing import Any


def build_report(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(question_results)
    pass_count = sum(1 for result in question_results if result["pass"])
    fail_count = total - pass_count
    average_latency_ms = _average([result.get("latency_ms", 0.0) for result in question_results])
    average_recall = _average([result.get("chunk_recall", 0.0) for result in question_results])
    average_keyword_coverage = _average(
        [result.get("keyword_coverage", 0.0) for result in question_results]
    )

    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "average_latency_ms": average_latency_ms,
        "average_recall": average_recall,
        "average_keyword_coverage": average_keyword_coverage,
        "questions": question_results,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(report: dict[str, Any]) -> None:
    print("Evaluation Summary")
    print("==========================")
    print(f"PASS\n{report['pass']}")
    print(f"FAIL\n{report['fail']}")
    print(f"Average Recall\n{report['average_recall']:.0%}")
    print(f"Average Keyword Coverage\n{report['average_keyword_coverage']:.0%}")
    print(f"Average Latency\n{report['average_latency_ms']:.0f}ms")
    print("==========================")


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
