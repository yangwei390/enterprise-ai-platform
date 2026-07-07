from abc import ABC, abstractmethod
from typing import Any, overload

from pydantic import BaseModel, Field


class RerankerError(Exception):
    pass


class RerankInputItem(BaseModel):
    id: str | None = None
    text: str
    metadata: dict = Field(default_factory=dict)
    original_score: float | None = None


class RerankResultItem(BaseModel):
    id: str | None = None
    index: int
    score: float
    metadata: dict = Field(default_factory=dict)


class RerankQuery(BaseModel):
    query: str
    chunks: list[Any]
    top_k: int = 5


class RerankedChunk(BaseModel):
    id: str
    original_score: float
    rerank_score: float
    text: str
    document_id: int | None
    knowledge_base_id: int | None
    chunk_index: int | None
    metadata: dict = Field(default_factory=dict)


class RerankResult(BaseModel):
    query: str
    top_k: int
    total: int
    chunks: list[RerankedChunk]
    metadata: dict = Field(default_factory=dict)


class BaseReranker(ABC):
    provider: str
    model_name: str

    @overload
    def rerank(
        self,
        query: str,
        items: list[RerankInputItem],
        top_k: int | None = None,
    ) -> list[RerankResultItem]: ...

    @overload
    def rerank(
        self,
        query: RerankQuery,
        items: None = None,
        top_k: None = None,
    ) -> RerankResult: ...

    @abstractmethod
    def rerank(
        self,
        query: str | RerankQuery,
        items: list[RerankInputItem] | None = None,
        top_k: int | None = None,
    ) -> list[RerankResultItem] | RerankResult:
        raise NotImplementedError

    def _rerank_query_compat(self, query: RerankQuery) -> RerankResult:
        input_items = [
            RerankInputItem(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                original_score=chunk.score,
            )
            for chunk in query.chunks
        ]
        result_items = self.rerank(
            query=query.query,
            items=input_items,
            top_k=query.top_k,
        )
        chunks_by_index = {index: chunk for index, chunk in enumerate(query.chunks)}
        reranked_chunks = []
        for rank, item in enumerate(result_items, start=1):
            chunk = chunks_by_index[item.index]
            metadata = {
                **chunk.metadata,
                **item.metadata,
                "rerank_score": item.score,
                "rerank_rank": rank,
                "rerank_provider": self.provider,
                "rerank_model": self.model_name,
            }
            reranked_chunks.append(
                RerankedChunk(
                    id=chunk.id,
                    original_score=chunk.score,
                    rerank_score=item.score,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    knowledge_base_id=chunk.knowledge_base_id,
                    chunk_index=chunk.chunk_index,
                    metadata=metadata,
                )
            )

        rerank_time_ms = 0.0
        if result_items:
            duration = result_items[0].metadata.get("duration_ms", 0.0)
            rerank_time_ms = float(duration) if isinstance(duration, int | float) else 0.0

        return RerankResult(
            query=query.query,
            top_k=query.top_k,
            total=len(reranked_chunks),
            chunks=reranked_chunks,
            metadata={
                "rerank_enabled": True,
                "rerank_provider": self.provider,
                "rerank_model": self.model_name,
                "rerank_top_k": query.top_k,
                "rerank_before_count": len(query.chunks),
                "rerank_after_count": len(reranked_chunks),
                "rerank_time_ms": rerank_time_ms,
                "rerank_failed": False,
                "rerank_error": None,
            },
        )
