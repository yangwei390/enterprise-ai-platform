from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class ToolSuccessMetric(BaseMetric):
    name = "tool_success"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        return result.error is None and bool(result.output.get("success", True)), {}


class ResultMatchMetric(BaseMetric):
    name = "result_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = case.expected.get("result")
        return result.output.get("result") == expected, {"expected": expected}


class ResultContainsMetric(BaseMetric):
    name = "result_contains"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = str(case.expected.get("contains", ""))
        return expected in str(result.output.get("result") or result.answer or ""), {}


class TimeoutMetric(BaseMetric):
    name = "timeout"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        return bool(result.metadata.get("timeout")), {}


class RetryCountMetric(BaseMetric):
    name = "retry_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return int(result.metadata.get("retry_count", 0)), {}


class CacheHitMetric(BaseMetric):
    name = "cache_hit"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        return bool(result.metadata.get("cache_hit")), {}


class MCPToolAvailableMetric(BaseMetric):
    name = "mcp_tool_available"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        return bool(result.metadata.get("mcp_tool_available")), {}


class MCPCallSuccessMetric(ToolSuccessMetric):
    name = "mcp_call_success"
