from backend.app.retrievers.base import RetrievedChunk
from pydantic import BaseModel, Field


class HybridRetrieveQuery(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5
    score_threshold: float | None = None
    metadata_filter: dict | None = None


class HybridRetrieveResult(BaseModel):
    chunks: list[RetrievedChunk]
    total: int
    metadata: dict = Field(default_factory=dict)
