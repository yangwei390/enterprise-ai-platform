from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class WorkflowCompletedMetric(BaseMetric):
    name = "workflow_completed"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        status = result.metadata.get("workflow_runtime", {}).get("status") or result.output.get(
            "status"
        )
        if status is None:
            status = "completed" if result.error is None else "failed"
        return status in {"completed", "success"}, {"status": status}


class WorkflowStatusMatchMetric(BaseMetric):
    name = "workflow_status_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = str(case.expected.get("status", "completed"))
        actual = str(result.output.get("status") or ("failed" if result.error else "completed"))
        return actual == expected, {"expected": expected, "actual": actual}


class NodePathMatchMetric(BaseMetric):
    name = "node_path_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = [str(item) for item in case.expected.get("node_path", [])]
        actual = [str(item.get("node_id") or item.get("name")) for item in result.trace]
        mode = str(case.expected.get("node_path_mode", "ordered_contains"))
        matched = actual == expected if mode == "exact" else _ordered_contains(actual, expected)
        return matched, {"expected": expected, "actual": actual, "mode": mode}


class NodeContainsMetric(BaseMetric):
    name = "node_contains"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = [str(item) for item in case.expected.get("nodes", [])]
        actual = [str(item.get("node_id") or item.get("name")) for item in result.trace]
        return all(item in actual for item in expected), {"actual": actual}


class StepCountMetric(BaseMetric):
    name = "step_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return len(result.trace), {}


class InterruptExpectedMetric(BaseMetric):
    name = "interrupt_expected"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = bool(case.expected.get("interrupt", False))
        actual = bool(result.output.get("interrupt") or result.metadata.get("interrupt"))
        return actual == expected, {"expected": expected, "actual": actual}


class FallbackUsedMetric(BaseMetric):
    name = "fallback_used"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        text = str(result.metadata)
        return "fallback" in text, {}


def _ordered_contains(actual: list[str], expected: list[str]) -> bool:
    pos = 0
    for item in actual:
        if pos < len(expected) and item == expected[pos]:
            pos += 1
    return pos == len(expected)
