from backend.app.context import ContextChunk
from backend.app.prompts import PromptMessage
from backend.app.rerankers import RerankedChunk
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5
    score_threshold: float | None = None
    metadata_filter: dict | None = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    total: int
    chunks: list[RerankedChunk]
    context_text: str
    context_total_chars: int
    context_chunks: list[ContextChunk]
    prompt_text: str
    prompt_messages: list[PromptMessage]
    answer: str | None
    llm_model: str | None
    llm_usage: dict | None
    metadata: dict = Field(default_factory=dict)
