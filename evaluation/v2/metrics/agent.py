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


class LoopIterationsMetric(BaseMetric):
    name = "loop_iterations"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return int(result.metadata.get("agent_loop", {}).get("loop_iterations", 0)), {}


class ReflectionCountMetric(BaseMetric):
    name = "reflection_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return int(result.metadata.get("reflection", {}).get("count", 0)), {}


class TerminationReasonMatchMetric(BaseMetric):
    name = "termination_reason_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = str(case.expected.get("termination_reason", "final_answer"))
        actual = str(result.metadata.get("agent_loop", {}).get("termination_reason"))
        return actual == expected, {"expected": expected, "actual": actual}


class RecommendationCountMetric(BaseMetric):
    name = "recommendation_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        maximum = int(case.expected.get("max_recommendations", 3))
        items = _tool_result_items(result, "recommend_products")
        return len(items) <= maximum, {"actual": len(items), "maximum": maximum}


class ConfirmationGuardMetric(BaseMetric):
    name = "confirmation_guard"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        write_tool_allowed = bool(case.expected.get("write_tool_allowed", False))
        called = any(
            (item.get("tool_name") or item.get("name")) == "create_after_sales_ticket"
            and item.get("status") != "blocked"
            for item in result.tool_calls
        )
        return write_tool_allowed or not called, {
            "write_tool_allowed": write_tool_allowed,
            "write_tool_called": called,
        }


class ManualDocumentScopeMetric(BaseMetric):
    name = "manual_document_scope"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected_document_id = case.expected.get("document_id")
        call_document_ids = [
            item.get("arguments", {}).get("document_id")
            for item in result.tool_calls
            if item.get("tool_name") == "knowledge_search"
            and isinstance(item.get("arguments"), dict)
        ]
        source_document_ids = [
            item.get("document_id")
            for item in result.sources
            if isinstance(item, dict)
        ]
        target_document_id = (
            expected_document_id
            if isinstance(expected_document_id, int)
            else call_document_ids[0]
            if len(call_document_ids) == 1 and isinstance(call_document_ids[0], int)
            else None
        )
        matched = target_document_id is not None and call_document_ids == [
            target_document_id
        ] and bool(source_document_ids) and all(
            document_id == target_document_id for document_id in source_document_ids
        )
        return matched, {
            "expected": target_document_id,
            "call_document_ids": call_document_ids,
            "source_document_ids": source_document_ids,
        }


class ProductConstraintMatchMetric(BaseMetric):
    name = "product_constraint_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        constraints = case.expected.get("product_constraints", {})
        if not isinstance(constraints, dict):
            return False, {"reason": "expected.product_constraints must be an object"}
        items = [
            item.get("product", item)
            for item in (
                _tool_result_items(result, "recommend_products")
                or _tool_result_items(result, "search_products")
            )
            if isinstance(item, dict)
        ]
        violations = [
            _product_constraint_violations(item, constraints)
            for item in items
        ]
        violations = [item for item in violations if item]
        return bool(items) and not violations, {
            "item_count": len(items),
            "violations": violations,
        }


def _tool_names(result: EvaluationTargetResult) -> list[str]:
    names = []
    for item in result.tool_calls:
        name = item.get("tool_name") or item.get("name")
        if name:
            names.append(str(name))
    return names


def _tool_result_items(
    result: EvaluationTargetResult,
    tool_name: str,
) -> list[dict]:
    for observation in reversed(result.observations):
        if observation.get("tool_name") != tool_name:
            continue
        raw_result = observation.get("raw_result")
        if isinstance(raw_result, dict) and isinstance(raw_result.get("items"), list):
            return [item for item in raw_result["items"] if isinstance(item, dict)]
    return []


def _product_constraint_violations(product: dict, constraints: dict) -> list[str]:
    violations: list[str] = []
    for field in ("category", "brand", "sale_status"):
        expected = constraints.get(field)
        if expected is not None and product.get(field) != expected:
            violations.append(field)
    price = product.get("price")
    if price is not None:
        numeric_price = float(price)
        if constraints.get("price_min") is not None and numeric_price < float(
            constraints["price_min"]
        ):
            violations.append("price_min")
        if constraints.get("price_max") is not None and numeric_price > float(
            constraints["price_max"]
        ):
            violations.append("price_max")
    if constraints.get("in_stock_only") and int(product.get("stock_quantity") or 0) <= 0:
        violations.append("stock_quantity")
    for field in ("required_features", "required_use_cases"):
        required = {str(item) for item in constraints.get(field, [])}
        actual_field = "features" if field == "required_features" else "use_cases"
        actual = {str(item) for item in product.get(actual_field, [])}
        if not required.issubset(actual):
            violations.append(field)
    return violations


def _ordered_contains(actual: list[str], expected: list[str]) -> bool:
    pos = 0
    for item in actual:
        if pos < len(expected) and item == expected[pos]:
            pos += 1
    return pos == len(expected)
