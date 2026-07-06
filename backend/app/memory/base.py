from pydantic import BaseModel, Field


class MemoryMessage(BaseModel):
    role: str
    content: str
    metadata: dict = Field(default_factory=dict)


class MemoryContext(BaseModel):
    summary: str | None = None
    recent_messages: list[MemoryMessage] = Field(default_factory=list)
    token_budget_used: int = 0
    metadata: dict = Field(default_factory=dict)
