from pydantic import BaseModel

from backend.app.retrievers import RetrievedChunk


class RetrieveRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    total: int
    chunks: list[RetrievedChunk]
