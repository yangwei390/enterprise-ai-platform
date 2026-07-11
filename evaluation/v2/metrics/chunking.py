from statistics import mean, median
from typing import Any

from backend.app.config.settings import settings

from evaluation.v2.metrics.base import BaseMetric
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class StructurePathCoverageMetric(BaseMetric):
    name = "structure_path_coverage"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        chunks = _chunks(result)
        if not chunks:
            return 0.0, {}
        covered = sum(1 for chunk in chunks if _metadata(chunk).get("section_path"))
        return covered / len(chunks), {"covered": covered, "total": len(chunks)}


class BoundaryIntegrityMetric(BaseMetric):
    name = "boundary_integrity"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        chunks = _chunks(result)
        if not chunks:
            return 0.0, {"proxy": True}
        broken = 0
        for chunk in chunks:
            metadata = _metadata(chunk)
            article_start = metadata.get("article_start")
            article_end = metadata.get("article_end")
            if (
                article_start is not None
                and article_end is not None
                and article_start != article_end
            ):
                broken += 1
        return 1 - (broken / len(chunks)), {"proxy": True, "broken": broken}


class ChapterPurityMetric(BaseMetric):
    name = "chapter_purity"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        chunks = _chunks(result)
        if not chunks:
            return 0.0, {"proxy": True}
        impure = 0
        for chunk in chunks:
            labels = _metadata(chunk).get("article_labels") or []
            text = str(_value(chunk, "text") or "")
            if "第" in text and "章" in text and len(labels) > 3:
                impure += 1
        return 1 - (impure / len(chunks)), {"proxy": True, "impure": impure}


class ArticleCoverageMetric(BaseMetric):
    name = "article_coverage"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        expected = {int(item) for item in case.expected.get("articles", [])}
        if not expected:
            return 1.0, {"reason": "no expected articles"}
        actual: set[int] = set()
        for chunk in _chunks(result):
            metadata = _metadata(chunk)
            start = metadata.get("article_start")
            end = metadata.get("article_end")
            if isinstance(start, int) and isinstance(end, int):
                actual.update(range(start, end + 1))
        return len(expected & actual) / len(expected), {
            "expected": sorted(expected),
            "actual": sorted(actual),
        }


class ParentChildLinkRateMetric(BaseMetric):
    name = "parent_child_link_rate"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        chunks = _chunks(result)
        child_chunks = [chunk for chunk in chunks if _metadata(chunk).get("chunk_role") == "child"]
        if not child_chunks:
            return 1.0, {"reason": "no child chunks"}
        linked = sum(1 for chunk in child_chunks if _metadata(chunk).get("parent_chunk_id"))
        return linked / len(child_chunks), {"linked": linked, "total": len(child_chunks)}


class ChunkLengthDistributionMetric(BaseMetric):
    name = "chunk_length_distribution"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[dict, dict]:
        lengths = [len(str(_value(chunk, "text") or "")) for chunk in _chunks(result)]
        if not lengths:
            return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0}, {}
        sorted_lengths = sorted(lengths)
        p95_index = min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95))
        return {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(mean(lengths), 2),
            "p50": median(lengths),
            "p95": sorted_lengths[p95_index],
        }, {}


class TinyChunkRateMetric(BaseMetric):
    name = "tiny_chunk_rate"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        chunks = _chunks(result)
        if not chunks:
            return 0.0, {}
        tiny = sum(
            1
            for chunk in chunks
            if len(str(_value(chunk, "text") or "")) < settings.CHUNK_MIN_CHARS
        )
        return tiny / len(chunks), {"tiny": tiny, "total": len(chunks)}


class OversizedChunkRateMetric(BaseMetric):
    name = "oversized_chunk_rate"

    def compute(self, case: EvaluationCase, result: EvaluationTargetResult) -> tuple[float, dict]:
        chunks = _chunks(result)
        if not chunks:
            return 0.0, {}
        limit = int(case.expected.get("max_chunk_chars") or settings.CHUNK_LEGAL_MAX_CHARS)
        oversized = sum(1 for chunk in chunks if len(str(_value(chunk, "text") or "")) > limit)
        return oversized / len(chunks), {
            "oversized": oversized,
            "total": len(chunks),
            "limit": limit,
        }


def _chunks(result: EvaluationTargetResult) -> list[Any]:
    return result.chunks or result.sources


def _metadata(chunk: Any) -> dict:
    metadata = _value(chunk, "metadata")
    return metadata if isinstance(metadata, dict) else {}


def _value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
