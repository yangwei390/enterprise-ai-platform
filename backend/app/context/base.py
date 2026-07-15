from abc import ABC, abstractmethod

from backend.app.rerankers import RerankedChunk
from pydantic import BaseModel, Field


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
    route_type: str | None = None
    strategy: str | None = None
    target_document_ids: list[int] = Field(default_factory=list)
    max_chunks: int = 12
    max_chars_per_document: int = 4000
    multi_document_diversity_enabled: bool = True
    multi_document_max_per_document: int = 4
    multi_document_min_documents: int = 2


class ContextBuildResult(BaseModel):
    query: str
    context_text: str
    chunks: list[ContextChunk]
    total_chunks: int
    total_chars: int
    selected_chunks: list[ContextChunk] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    document_groups: list[dict] = Field(default_factory=list)
    truncated: bool = False
    deduplicated_count: int = 0
    merged_count: int = 0
    skipped_count: int = 0
    metadata: dict = Field(default_factory=dict)


class BaseContextBuilder(ABC):
    @abstractmethod
    def build(self, request: ContextBuildRequest) -> ContextBuildResult:
        raise NotImplementedError
