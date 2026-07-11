from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class KeywordCoverageMetric(BaseMetric):
    name = "keyword_coverage"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        keywords = [str(item) for item in case.expected.get("keywords", [])]
        if not keywords:
            return 1.0, {}
        scope = str(case.input.get("evaluation_scope", "answer"))
        text = _scope_text(scope, result)
        matched = [keyword for keyword in keywords if keyword in text]
        return len(matched) / len(keywords), {"matched": matched, "scope": scope}


class ExactMatchMetric(BaseMetric):
    name = "exact_match"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = str(case.expected.get("answer", ""))
        return (result.answer or "") == expected, {"expected": expected}


class ContainsKeywordsMetric(KeywordCoverageMetric):
    name = "contains_keywords"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        coverage, details = super().compute(case, result)
        return coverage >= 1.0, details


class AnswerLengthMetric(BaseMetric):
    name = "answer_length"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return len(result.answer or ""), {}


class CitationCountMetric(BaseMetric):
    name = "citation_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return len(result.citations), {}


class SourceCoverageMetric(BaseMetric):
    name = "source_coverage"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        expected = [str(item) for item in case.expected.get("sources", [])]
        if not expected:
            return 1.0, {}
        actual = [str(item.get("source")) for item in result.sources if isinstance(item, dict)]
        matched = [item for item in expected if item in actual]
        return len(matched) / len(expected), {"actual": actual}


class EmptyAnswerMetric(BaseMetric):
    name = "empty_answer"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        return not bool((result.answer or "").strip()), {}


def _scope_text(scope: str, result: EvaluationTargetResult) -> str:
    if scope == "chunks":
        return "\n".join(str(item.get("text", item)) for item in result.chunks)
    if scope == "context":
        return str(result.output.get("context_text", ""))
    return result.answer or ""
