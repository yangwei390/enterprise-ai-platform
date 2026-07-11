from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class ToolSelectionAccuracyMetric(BaseMetric):
    name = "tool_selection_accuracy"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        expected = [str(item) for item in case.expected.get("tools", [])]
        if not expected:
            return 1.0, {}
        actual = _tool_names(result)
        matched = [tool for tool in expected if tool in actual]
        return len(matched) / len(expected), {"expected": expected, "actual": actual}


class ToolCallSuccessRateMetric(BaseMetric):
    name = "tool_call_success_rate"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        observations = result.observations
        if not observations:
            return 1.0 if not case.expected.get("tools") else 0.0, {}
        successes = sum(1 for item in observations if item.get("success", True))
        return successes / len(observations), {"total": len(observations)}


class ToolSequenceMatchMetric(BaseMetric):
    name = "tool_sequence_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = [str(item) for item in case.expected.get("tool_sequence", [])]
        actual = _tool_names(result)
        mode = str(case.expected.get("tool_sequence_mode", "ordered_contains"))
        if mode == "exact":
            matched = actual == expected
        elif mode == "contains":
            matched = all(item in actual for item in expected)
        else:
            matched = _ordered_contains(actual, expected)
        return matched, {"expected": expected, "actual": actual, "mode": mode}


class UnnecessaryToolCallsMetric(BaseMetric):
    name = "unnecessary_tool_calls"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        expected = set(str(item) for item in case.expected.get("tools", []))
        actual = _tool_names(result)
        return len([tool for tool in actual if tool not in expected]), {"actual": actual}


class AgentStepCountMetric(BaseMetric):
    name = "agent_step_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return len(result.trace), {}


class FinalAnswerKeywordCoverageMetric(BaseMetric):
    name = "final_answer_keyword_coverage"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        from evaluation.v2.metrics.generation import KeywordCoverageMetric

        return KeywordCoverageMetric().compute(case, result)


def _tool_names(result: EvaluationTargetResult) -> list[str]:
    names = []
    for item in result.tool_calls:
        name = item.get("tool_name") or item.get("name")
        if name:
            names.append(str(name))
    return names


def _ordered_contains(actual: list[str], expected: list[str]) -> bool:
    pos = 0
    for item in actual:
        if pos < len(expected) and item == expected[pos]:
            pos += 1
    return pos == len(expected)
