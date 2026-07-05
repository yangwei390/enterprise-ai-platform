from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class RetrieveQuery(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5


class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str
    document_id: int | None
    knowledge_base_id: int | None
    chunk_index: int | None
    metadata: dict = Field(default_factory=dict)


class RetrieveResult(BaseModel):
    query: str
    top_k: int
    total: int
    chunks: list[RetrievedChunk]
    metadata: dict = Field(default_factory=dict)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: RetrieveQuery) -> RetrieveResult:
        raise NotImplementedError
