import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.config.settings import settings

from evaluation.v2.schemas import EvaluationCaseResult, EvaluationReportV2

SENSITIVE_KEYS = ("api_key", "authorization", "cookie", "token", "password", "secret")


def build_report(
    *,
    run_id: str,
    suite_id: str,
    suite_name: str,
    suite_version: str,
    started_at: datetime,
    finished_at: datetime,
    cases: list[EvaluationCaseResult],
    regression_summary: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReportV2:
    total = len(cases)
    skipped = sum(1 for case in cases if case.status == "skipped")
    passed = sum(1 for case in cases if case.passed and case.status != "skipped")
    failed = total - passed - skipped
    target_counts = Counter(case.target for case in cases)
    metric_values: dict[str, list[float]] = defaultdict(list)
    usage_summary = {"available": False, "total_tokens": None}
    for case in cases:
        for metric in case.metrics:
            if isinstance(metric.value, int | float):
                metric_values[metric.name].append(float(metric.value))
    metric_summary = {
        name: {"average": sum(values) / len(values), "count": len(values)}
        for name, values in metric_values.items()
        if values
    }
    return EvaluationReportV2(
        run_id=run_id,
        suite_id=suite_id,
        suite_name=suite_name,
        suite_version=suite_version,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_ms=round((finished_at - started_at).total_seconds() * 1000, 2),
        passed=failed == 0,
        pass_rate=(passed / (total - skipped)) if total != skipped else 1.0,
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        skipped_cases=skipped,
        target_summary=dict(target_counts),
        metric_summary=metric_summary,
        usage_summary=usage_summary,
        cost_summary={"estimated_cost": None, "currency": None, "available": False},
        regression_summary=regression_summary or {},
        cases=cases,
        environment={
            "app_env": settings.APP_ENV,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "agent_runtime": settings.AGENT_RUNTIME,
            "workflow_runtime": settings.WORKFLOW_RUNTIME,
        },
        config={
            "max_concurrency": settings.EVALUATION_MAX_CONCURRENCY,
            "llm_judge_enabled": settings.EVALUATION_LLM_JUDGE_ENABLED,
        },
        metadata=metadata or {},
    )


def write_report(report: EvaluationReportV2, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(redact_and_truncate(report.model_dump()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def redact_and_truncate(value: Any, max_chars: int | None = None) -> Any:
    max_len = max_chars or settings.EVALUATION_REPORT_MAX_TEXT_CHARS
    if isinstance(value, dict):
        return {
            key: "***REDACTED***" if _is_sensitive(key) else redact_and_truncate(item, max_len)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_and_truncate(item, max_len) for item in value]
    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + "...[truncated]"
    return value


def now_utc() -> datetime:
    return datetime.now(UTC)


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(item in lowered for item in SENSITIVE_KEYS)
