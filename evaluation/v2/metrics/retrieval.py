from typing import Any

from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class RetrieverHitMetric(BaseMetric):
    name = "retriever_hit"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[bool, dict]:
        expected = _expected_documents(case)
        if not expected:
            return True, {"reason": "no expected documents"}
        actual = [_source_name(item) for item in result.sources or result.chunks]
        hit = any(
            expected_doc
            and actual_doc
            and (expected_doc in actual_doc or actual_doc in expected_doc)
            for expected_doc in expected
            for actual_doc in actual
        )
        return hit, {"expected": expected, "actual": actual}


class ChunkRecallMetric(BaseMetric):
    name = "chunk_recall"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        expected = _expected_chunks(case)
        if not expected:
            return 1.0, {"reason": "no expected chunks"}
        actual = {
            value
            for value in (_chunk_index(item) for item in result.sources or result.chunks)
            if value is not None
        }
        matched = set(expected) & actual
        return len(matched) / len(set(expected)), {"expected": expected, "actual": sorted(actual)}


class MRRMetric(BaseMetric):
    name = "mrr"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        expected = set(_expected_chunks(case))
        if not expected:
            return 1.0, {}
        for index, item in enumerate(result.sources or result.chunks, start=1):
            if _chunk_index(item) in expected:
                return 1 / index, {"rank": index}
        return 0.0, {"rank": None}


class HitRateAtKMetric(BaseMetric):
    name = "hit_rate_at_k"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        expected_docs = set(_expected_documents(case))
        expected_chunks = set(_expected_chunks(case))
        k = (
            int(case.thresholds.get("hit_rate_at_k", {}).get("k", 5))
            if isinstance(case.thresholds.get("hit_rate_at_k"), dict)
            else 5
        )
        top_items = (result.sources or result.chunks)[:k]
        hit = any(
            _source_name(item) in expected_docs or _chunk_index(item) in expected_chunks
            for item in top_items
        )
        return 1.0 if hit else 0.0, {"k": k}


class ContextPrecisionProxyMetric(BaseMetric):
    name = "context_precision_proxy"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        keywords = _expected_keywords(case)
        if not keywords:
            return 1.0, {"proxy": True}
        chunks = result.chunks or result.sources
        if not chunks:
            return 0.0, {"proxy": True}
        relevant = sum(
            1
            for chunk in chunks
            if any(keyword in str(_value(chunk, "text") or chunk) for keyword in keywords)
        )
        return relevant / len(chunks), {"proxy": True, "chunk_count": len(chunks)}


class RetrievedCountMetric(BaseMetric):
    name = "retrieved_count"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[int, dict]:
        return len(result.sources or result.chunks), {}


def _expected_documents(case: EvaluationCase) -> list[str]:
    return [
        str(item)
        for item in (
            case.expected.get("documents")
            or case.expected.get("expected_documents")
            or case.expected.get("document_ids")
            or []
        )
    ]


def _expected_chunks(case: EvaluationCase) -> list[int]:
    return [
        int(item)
        for item in (
            case.expected.get("chunks")
            or case.expected.get("chunk_indexes")
            or case.expected.get("expected_chunks")
            or []
        )
    ]


def _expected_keywords(case: EvaluationCase) -> list[str]:
    return [str(item) for item in case.expected.get("keywords", [])]


def _source_name(item: Any) -> str | None:
    value = _value(item, "source")
    metadata = _value(item, "metadata")
    if value:
        return str(value)
    if isinstance(metadata, dict) and metadata.get("source"):
        return str(metadata["source"])
    document_id = _value(item, "document_id")
    return str(document_id) if document_id is not None else None


def _chunk_index(item: Any) -> int | None:
    value = _value(item, "chunk_index")
    if isinstance(value, int):
        return value
    metadata = _value(item, "metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("chunk_index"), int):
        return metadata["chunk_index"]
    return None


def _value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
