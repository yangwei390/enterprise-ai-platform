from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    document_id: int | None = None
    knowledge_base_id: int | None = None
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    token_count: int | None = None
    metadata: dict = Field(default_factory=dict)


class ChunkResult(BaseModel):
    strategy: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunks: list[Chunk]
    total_chunks: int
    total_tokens: int | None = None
    metadata: dict = Field(default_factory=dict)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> ChunkResult:
        raise NotImplementedError
