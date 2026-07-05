from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    id: str
    vector: list[float]
    text: str
    document_id: int | None
    knowledge_base_id: int | None
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class VectorStoreResult(BaseModel):
    collection_name: str
    total_records: int
    ids: list[str]
    metadata: dict = Field(default_factory=dict)


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> VectorStoreResult:
        raise NotImplementedError
