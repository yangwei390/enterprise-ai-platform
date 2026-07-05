from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from backend.app.chunkers import Chunk


class EmbeddingItem(BaseModel):
    chunk_index: int
    text: str
    vector: list[float]
    document_id: int | None
    knowledge_base_id: int | None
    metadata: dict = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    items: list[EmbeddingItem]
    total_items: int
    model_name: str
    dimension: int
    metadata: dict = Field(default_factory=dict)


class BaseEmbedding(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingResult:
        raise NotImplementedError
