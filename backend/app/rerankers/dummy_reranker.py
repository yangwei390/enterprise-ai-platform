from backend.app.rerankers.base import (
    BaseReranker,
    RerankInputItem,
    RerankQuery,
    RerankResult,
    RerankResultItem,
)


class DummyReranker(BaseReranker):
    provider = "dummy"
    model_name = "dummy-reranker"

    def rerank(
        self,
        query: str | RerankQuery,
        items: list[RerankInputItem] | None = None,
        top_k: int | None = None,
    ) -> list[RerankResultItem] | RerankResult:
        if isinstance(query, RerankQuery):
            return self._rerank_query_compat(query)

        if items is None:
            return []

        limit = top_k or len(items)
        result_items = []
        for index, item in enumerate(items[:limit]):
            score = item.original_score
            if score is None:
                score = float(len(items) - index)
            result_items.append(
                RerankResultItem(
                    id=item.id,
                    index=index,
                    score=score,
                    metadata={
                        "provider": self.provider,
                        "model": self.model_name,
                    },
                )
            )
        return result_items
