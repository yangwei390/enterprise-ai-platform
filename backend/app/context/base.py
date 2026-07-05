from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from backend.app.rerankers import RerankedChunk


class ContextChunk(BaseModel):
    id: str
    text: str
    document_id: int | None
    knowledge_base_id: int | None
    chunk_index: int | None
    score: float | None
    source: str | None
    metadata: dict = Field(default_factory=dict)


class ContextBuildRequest(BaseModel):
    query: str
    chunks: list[RerankedChunk]
    max_context_chars: int = 8000


class ContextBuildResult(BaseModel):
    query: str
    context_text: str
    chunks: list[ContextChunk]
    total_chunks: int
    total_chars: int
    metadata: dict = Field(default_factory=dict)


class BaseContextBuilder(ABC):
    @abstractmethod
    def build(self, request: ContextBuildRequest) -> ContextBuildResult:
        raise NotImplementedError
