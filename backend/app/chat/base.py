from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    top_k: int = 5
    score_threshold: float | None = None


class ChatSource(BaseModel):
    id: str
    text: str
    document_id: int | None
    knowledge_base_id: int | None
    chunk_index: int | None
    score: float | None
    source: str | None
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[ChatSource]
    context_text: str
    prompt_text: str
    llm_model: str | None = None
    metadata: dict = Field(default_factory=dict)
