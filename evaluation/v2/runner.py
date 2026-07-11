import asyncio
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from backend.app.config.settings import settings

from evaluation.v2.baseline import BaselineManager
from evaluation.v2.history import HistoryManager
from evaluation.v2.metrics import get_metric
from evaluation.v2.regression import RegressionComparator
from evaluation.v2.report import build_report, write_report
from evaluation.v2.schemas import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationContext,
    EvaluationReportV2,
    EvaluationSuite,
    EvaluationTargetResult,
    MetricResult,
)
from evaluation.v2.suite import load_suite
from evaluation.v2.targets import get_target


class EvaluationRunnerV2:
    def __init__(
        self,
        concurrency: int | None = None,
        fail_fast: bool | None = None,
    ) -> None:
        self.concurrency = max(1, concurrency or settings.EVALUATION_MAX_CONCURRENCY)
        self.fail_fast = settings.EVALUATION_FAIL_FAST if fail_fast is None else fail_fast

    async def arun_suite(
        self,
        suite_or_path: EvaluationSuite | str | Path,
        case_ids: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        baseline: str | None = None,
        compare: bool = False,
        save_baseline: str | None = None,
        output: str | Path | None = None,
    ) -> EvaluationReportV2:
        suite = (
            load_suite(suite_or_path)
            if not isinstance(suite_or_path, EvaluationSuite)
            else suite_or_path
        )
        run_id = str(uuid.uuid4())
        context = EvaluationContext(
            run_id=run_id,
            suite_id=suite.id,
            metadata={
                "started_at": _utc_now(),
                "evaluation_version": "2.0",
            },
        )
        selected_cases = self._select_cases(suite, set(case_ids or []), set(tags or []))
        started_at = datetime.now(UTC)
        semaphore = asyncio.Semaphore(max(1, suite.concurrency or self.concurrency))
        results: list[EvaluationCaseResult] = []

        async def run_one(case: EvaluationCase) -> EvaluationCaseResult:
            async with semaphore:
                return await self.arun_case(case, suite, context)

        if self.fail_fast:
            for case in selected_cases:
                result = await run_one(case)
                results.append(result)
                if result.status == "failed":
                    break
        else:
            results = await asyncio.gather(*(run_one(case) for case in selected_cases))

        report = build_report(
            run_id=run_id,
            suite_id=suite.id,
            suite_name=suite.name,
            suite_version=suite.version,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            cases=results,
        )
        if compare and baseline:
            baseline_report = BaselineManager().load_baseline(baseline)
            report.regression_summary = RegressionComparator().compare(
                report.model_dump(mode="json"),
                baseline_report.model_dump(mode="json"),
            )

        HistoryManager().save(report)
        if save_baseline:
            BaselineManager().save_baseline(save_baseline, report)
        if output:
            write_report(report, Path(output))
        return report

    def run_suite(
        self,
        suite_or_path: EvaluationSuite | str | Path,
        case_ids: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        baseline: str | None = None,
        compare: bool = False,
        save_baseline: str | None = None,
        output: str | Path | None = None,
    ) -> EvaluationReportV2:
        return asyncio.run(
            self.arun_suite(
                suite_or_path,
                case_ids=case_ids,
                tags=tags,
                baseline=baseline,
                compare=compare,
                save_baseline=save_baseline,
                output=output,
            )
        )

    async def arun_case(
        self,
        case: EvaluationCase,
        suite: EvaluationSuite | None = None,
        context: EvaluationContext | None = None,
    ) -> EvaluationCaseResult:
        suite = suite or EvaluationSuite(id="single", name="single", cases=[case])
        context = context or EvaluationContext(run_id=str(uuid.uuid4()), suite_id=suite.id)

        if not case.enabled:
            return self._skipped_case_result(case, "case disabled")

        missing_requirement = self._missing_requirement(case)
        if missing_requirement:
            return self._skipped_case_result(case, missing_requirement)

        timeout = (
            case.timeout_seconds
            or suite.timeout_seconds
            or settings.EVALUATION_DEFAULT_TIMEOUT_SECONDS
        )
        try:
            async with asyncio.timeout(timeout):
                target_result = await get_target(case.target).arun(case, context)
                metrics = await self._evaluate_metrics(case, suite, target_result, context)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            return EvaluationCaseResult(
                case_id=case.id,
                target=case.target,
                status="failed",
                passed=False,
                duration_ms=0,
                metrics=[],
                target_result_summary={},
                error=f"case timeout after {timeout}s: {exc}",
            )
        except Exception as exc:
            return EvaluationCaseResult(
                case_id=case.id,
                target=case.target,
                status="failed",
                passed=False,
                duration_ms=0,
                metrics=[],
                target_result_summary={},
                error=str(exc),
            )

        if target_result.skipped:
            return EvaluationCaseResult(
                case_id=case.id,
                target=case.target,
                status="skipped",
                passed=False,
                duration_ms=target_result.duration_ms,
                metrics=[],
                target_result_summary={"skip_reason": target_result.skip_reason},
                error=target_result.error,
            )

        passed = target_result.error is None and all(metric.passed for metric in metrics)
        return EvaluationCaseResult(
            case_id=case.id,
            target=case.target,
            status="passed" if passed else "failed",
            passed=passed,
            duration_ms=target_result.duration_ms,
            metrics=metrics,
            target_result_summary=self._summarize_target_result(target_result),
            error=target_result.error,
            metadata={"target_metadata": target_result.metadata},
        )

    def run_case(
        self,
        case: EvaluationCase,
        suite: EvaluationSuite | None = None,
        context: EvaluationContext | None = None,
    ) -> EvaluationCaseResult:
        return asyncio.run(self.arun_case(case, suite, context))

    async def _evaluate_metrics(
        self,
        case: EvaluationCase,
        suite: EvaluationSuite,
        target_result: EvaluationTargetResult,
        context: EvaluationContext,
    ) -> list[MetricResult]:
        metrics = self._case_metrics(case, suite)
        metric_results: list[MetricResult] = []
        merged_thresholds = {**suite.default_thresholds, **case.thresholds}
        metric_case = case.model_copy(update={"thresholds": merged_thresholds})
        for metric_name in metrics:
            metric = get_metric(metric_name)
            metric_results.append(await metric.evaluate(metric_case, target_result, context))
        return metric_results

    def _case_metrics(self, case: EvaluationCase, suite: EvaluationSuite) -> list[str]:
        names = list(dict.fromkeys([*suite.default_metrics, *case.metrics]))
        return names or ["latency_ms"]

    def _select_cases(
        self,
        suite: EvaluationSuite,
        case_ids: set[str],
        tags: set[str],
    ) -> list[EvaluationCase]:
        cases = suite.cases
        if case_ids:
            cases = [case for case in cases if case.id in case_ids]
        if tags:
            cases = [case for case in cases if tags.intersection(case.tags)]
        return cases

    def _missing_requirement(self, case: EvaluationCase) -> str | None:
        for requirement in case.requires:
            if requirement == "mcp" and not settings.MCP_ENABLED:
                return "requires MCP, but MCP_ENABLED=false"
            if requirement == "workflow_v2" and not settings.WORKFLOW_V2_ENABLED:
                return "requires Workflow V2, but WORKFLOW_V2_ENABLED=false"
        return None

    def _skipped_case_result(self, case: EvaluationCase, reason: str) -> EvaluationCaseResult:
        return EvaluationCaseResult(
            case_id=case.id,
            target=case.target,
            status="skipped",
            passed=False,
            duration_ms=0,
            metrics=[],
            target_result_summary={"skip_reason": reason},
            error=None,
        )

    def _summarize_target_result(self, result: EvaluationTargetResult) -> dict:
        return {
            "answer": result.answer,
            "source_count": len(result.sources),
            "citation_count": len(result.citations),
            "chunk_count": len(result.chunks),
            "tool_call_count": len(result.tool_calls),
            "trace_count": len(result.trace),
            "usage": result.usage,
            "duration_ms": result.duration_ms,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["EvaluationRunnerV2"]
