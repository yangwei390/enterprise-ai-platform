from time import perf_counter

from backend.app.retrievers.document_router import RouteType, RoutingResult
from backend.app.retrievers.strategy.models import (
    RetrievalStrategy,
    RetrievalStrategyType,
)


class StrategySelector:
    def select(
        self,
        *,
        routing_result: RoutingResult | None,
        requested_top_k: int,
        max_top_k: int,
        per_document_top_k: int,
    ) -> RetrievalStrategy:
        started = perf_counter()
        global_budget = self._global_budget(requested_top_k, max_top_k)
        if routing_result is None:
            return self._strategy(
                strategy="GLOBAL",
                route_type="UNKNOWN",
                document_ids=[],
                global_budget=global_budget,
                per_document_budget=None,
                fallback=True,
                reason="missing_routing_result",
                started=started,
            )

        document_ids = list(dict.fromkeys(routing_result.target_document_ids))
        if routing_result.route_type == "DOCUMENT" and document_ids:
            return self._strategy(
                strategy="DOCUMENT",
                route_type=routing_result.route_type,
                document_ids=document_ids,
                global_budget=global_budget,
                per_document_budget=global_budget,
                fallback=False,
                reason="document_route",
                started=started,
            )

        if routing_result.route_type == "MULTI_DOCUMENT" and document_ids:
            return self._strategy(
                strategy="MULTI_DOCUMENT",
                route_type=routing_result.route_type,
                document_ids=document_ids,
                global_budget=global_budget,
                per_document_budget=self._per_document_budget(
                    document_count=len(document_ids),
                    global_budget=global_budget,
                    configured_budget=per_document_top_k,
                ),
                fallback=False,
                reason="multi_document_route",
                started=started,
            )

        if routing_result.route_type == "UNKNOWN":
            return self._strategy(
                strategy="GLOBAL",
                route_type=routing_result.route_type,
                document_ids=[],
                global_budget=global_budget,
                per_document_budget=None,
                fallback=True,
                reason="unknown_route_fallback_to_global",
                started=started,
            )

        return self._strategy(
            strategy="GLOBAL",
            route_type=routing_result.route_type,
            document_ids=[],
            global_budget=global_budget,
            per_document_budget=None,
            fallback=False,
            reason="knowledge_base_global_route",
            started=started,
        )

    def _strategy(
        self,
        *,
        strategy: RetrievalStrategyType,
        route_type: RouteType,
        document_ids: list[int],
        global_budget: int,
        per_document_budget: int | None,
        fallback: bool,
        reason: str,
        started: float,
    ) -> RetrievalStrategy:
        return RetrievalStrategy(
            strategy=strategy,
            route_type=route_type,
            document_ids=document_ids,
            global_budget=global_budget,
            per_document_budget=per_document_budget,
            fallback=fallback,
            metadata={
                "reason": reason,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )

    def _global_budget(self, requested_top_k: int, max_top_k: int) -> int:
        if max_top_k <= 0:
            return max(1, requested_top_k)
        return max(1, min(requested_top_k, max_top_k))

    def _per_document_budget(
        self,
        *,
        document_count: int,
        global_budget: int,
        configured_budget: int,
    ) -> int:
        if document_count <= 0:
            return global_budget
        fair_budget = max(1, global_budget // document_count)
        if configured_budget <= 0:
            return fair_budget
        return max(1, min(configured_budget, fair_budget))
