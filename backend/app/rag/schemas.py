from backend.app.chat.base import ChatSource, CitationItem
from backend.app.llms import LLMMessage
from pydantic import BaseModel, Field


class RagChatInput(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    conversation_id: int | None = None
    memory_context: str | None = None
    memory_messages: list[LLMMessage] = Field(default_factory=list)
    history_messages: list[dict] = Field(default_factory=list)
    model: str | None = None
    top_k: int = 5
    score_threshold: float | None = None
    metadata_filter: dict | None = None
    stream: bool = False
    metadata: dict = Field(default_factory=dict)


class RagChatResult(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    citations: list[CitationItem] = Field(default_factory=list)
    context_text: str
    prompt_text: str
    llm_model: str | None = None
    metadata: dict = Field(default_factory=dict)
