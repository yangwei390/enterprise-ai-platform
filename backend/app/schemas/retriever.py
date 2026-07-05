from pydantic import BaseModel

from backend.app.context import ContextChunk
from backend.app.rerankers import RerankedChunk


class RetrieveRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    total: int
    chunks: list[RerankedChunk]
    context_text: str
    context_total_chars: int
    context_chunks: list[ContextChunk]
