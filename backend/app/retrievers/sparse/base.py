from backend.app.retrievers.planning import RetrievalConstraint
from pydantic import BaseModel, Field


class SparseDocument(BaseModel):
    id: str
    text: str
    document_id: int | None = None
    knowledge_base_id: int | None = None
    chunk_index: int | None = None
    metadata: dict = Field(default_factory=dict)


class SparseSearchQuery(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5
    metadata_filter: dict | None = None
    constraints: list[RetrievalConstraint] = Field(default_factory=list)


class SparseSearchResult(BaseModel):
    id: str
    score: float
    text: str
    document_id: int | None = None
    knowledge_base_id: int | None = None
    chunk_index: int | None = None
    metadata: dict = Field(default_factory=dict)
