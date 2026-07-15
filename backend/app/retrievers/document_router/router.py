from time import perf_counter

from backend.app.query import QueryRewriteResult
from backend.app.retrievers.document_router.models import RoutingResult
from backend.app.retrievers.metadata_filter import AutoMetadataFilterResult
from backend.app.retrievers.query_understanding import QueryUnderstandingResult


class DocumentRouter:
    def route(
        self,
        *,
        understanding: QueryUnderstandingResult | None,
        rewrite_result: QueryRewriteResult | None,
        metadata_filter_result: AutoMetadataFilterResult | None,
        max_candidates: int,
    ) -> RoutingResult:
        started = perf_counter()
        candidate_ids = self._candidate_ids(metadata_filter_result, max_candidates)
        document_hints = understanding.document_hints if understanding is not None else []
        comparison_targets = (
            understanding.comparison_targets if understanding is not None else []
        )
        intent = understanding.intent if understanding is not None else None
        normalized_query = understanding.normalized_query if understanding is not None else ""

        result = self._decide(
            candidate_ids=candidate_ids,
            document_hints=document_hints,
            comparison_targets=comparison_targets,
            intent=intent,
            normalized_query=normalized_query,
        )
        result.metadata.update(
            {
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "document_hints": document_hints,
                "comparison_targets": comparison_targets,
                "intent": intent,
                "rewrite_type": (
                    rewrite_result.rewrite_type if rewrite_result is not None else None
                ),
                "metadata_filter_applied": (
                    metadata_filter_result.auto_filter_applied
                    if metadata_filter_result is not None
                    else False
                ),
            }
        )
        return result

    def _decide(
        self,
        *,
        candidate_ids: list[int],
        document_hints: list[str],
        comparison_targets: list[str],
        intent: str | None,
        normalized_query: str,
    ) -> RoutingResult:
        if not normalized_query and not candidate_ids and not document_hints:
            return RoutingResult(
                route_type="UNKNOWN",
                confidence=0.0,
                reason="insufficient_query_information",
            )

        if candidate_ids and intent in ("comparison", "multi_document"):
            return RoutingResult(
                route_type="MULTI_DOCUMENT",
                target_document_ids=candidate_ids,
                candidate_document_ids=candidate_ids,
                confidence=0.85,
                reason="multi_document_query_with_candidates",
            )

        if len(candidate_ids) == 1:
            return RoutingResult(
                route_type="DOCUMENT",
                target_document_ids=candidate_ids,
                candidate_document_ids=candidate_ids,
                confidence=0.9,
                reason="document_identity_match",
            )

        if candidate_ids and document_hints:
            return RoutingResult(
                route_type="DOCUMENT",
                target_document_ids=candidate_ids,
                candidate_document_ids=candidate_ids,
                confidence=0.75,
                reason="document_hints_matched_candidates",
            )

        if document_hints and not candidate_ids:
            return RoutingResult(
                route_type="UNKNOWN",
                candidate_document_ids=[],
                confidence=0.2,
                reason="document_hints_without_candidates",
                metadata={"unmatched_document_hints": document_hints},
            )

        if intent in ("comparison", "multi_document") or len(comparison_targets) >= 2:
            return RoutingResult(
                route_type="MULTI_DOCUMENT",
                candidate_document_ids=candidate_ids,
                target_document_ids=candidate_ids,
                confidence=0.55,
                reason="multi_document_query_without_candidates",
            )

        return RoutingResult(
            route_type="KNOWLEDGE_BASE",
            candidate_document_ids=candidate_ids,
            confidence=0.6,
            reason="no_document_scope_detected",
        )

    def _candidate_ids(
        self,
        metadata_filter_result: AutoMetadataFilterResult | None,
        max_candidates: int,
    ) -> list[int]:
        if metadata_filter_result is None:
            return []
        ids = list(dict.fromkeys(metadata_filter_result.candidate_document_ids))
        if max_candidates > 0:
            return ids[:max_candidates]
        return ids
