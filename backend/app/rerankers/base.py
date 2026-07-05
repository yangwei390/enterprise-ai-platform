from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from backend.app.retrievers import RetrievedChunk


class RerankQuery(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
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
    @abstractmethod
    def rerank(self, query: RerankQuery) -> RerankResult:
        raise NotImplementedError
