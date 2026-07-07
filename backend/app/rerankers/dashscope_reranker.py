import time
from typing import Any

import requests
from backend.app.rerankers.base import (
    BaseReranker,
    RerankerError,
    RerankInputItem,
    RerankQuery,
    RerankResult,
    RerankResultItem,
)
from backend.app.rerankers.config import RerankerConfig


class DashScopeReranker(BaseReranker):
    provider = "dashscope"

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.model_name = config.model

    def rerank(
        self,
        query: str | RerankQuery,
        items: list[RerankInputItem] | None = None,
        top_k: int | None = None,
    ) -> list[RerankResultItem] | RerankResult:
        if isinstance(query, RerankQuery):
            try:
                return self._rerank_query_compat(query)
            except Exception as exc:
                if not self.config.fail_open:
                    raise
                from backend.app.rerankers.dummy_reranker import DummyReranker

                fallback_result = DummyReranker().rerank(query)
                if not isinstance(fallback_result, RerankResult):
                    raise RerankerError(
                        "Dummy reranker fallback returned invalid result"
                    ) from exc
                fallback_result.metadata.update(
                    {
                        "rerank_failed": True,
                        "rerank_error": str(exc),
                        "rerank_fail_open": True,
                        "requested_provider": self.provider,
                        "requested_model": self.model_name,
                    }
                )
                return fallback_result

        if items is None or not items:
            return []
        if not self.config.api_key:
            raise RerankerError("DashScope Rerank API Key未配置")

        started_at = time.perf_counter()
        endpoint = self._build_endpoint()
        documents = [item.text for item in items]
        body: dict[str, Any] = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
            },
        }
        if top_k is not None:
            body["parameters"]["top_n"] = top_k

        response = requests.post(
            endpoint,
            json=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )
        if response.status_code >= 400:
            raise RerankerError(
                f"DashScope rerank failed | status={response.status_code} | "
                f"response={response.text}"
            )

        payload = response.json()
        raw_results = payload.get("output", {}).get("results", [])
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        result_items = []
        for raw_result in raw_results:
            index = raw_result.get("index")
            score = raw_result.get("relevance_score")
            if not isinstance(index, int) or not isinstance(score, int | float):
                continue
            item = items[index]
            result_items.append(
                RerankResultItem(
                    id=item.id,
                    index=index,
                    score=float(score),
                    metadata={
                        "provider": self.provider,
                        "model": self.model_name,
                        "duration_ms": duration_ms,
                    },
                )
            )
        return result_items[:top_k] if top_k is not None else result_items

    def _build_endpoint(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if "/services/rerank" in base_url:
            return base_url
        if "/compatible-mode" in base_url:
            root = base_url.split("/compatible-mode", maxsplit=1)[0]
            return f"{root}/api/v1/services/rerank/text-rerank/text-rerank"
        return f"{base_url}/api/v1/services/rerank/text-rerank/text-rerank"
