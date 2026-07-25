from pydantic import BaseModel, Field


class CalculatorArgs(BaseModel):
    expression: str


class EchoArgs(BaseModel):
    text: str


class CurrentTimeArgs(BaseModel):
    timezone: str | None = None


class KnowledgeSearchArgs(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    document_id: int | None = Field(default=None, ge=1)
    conversation_id: int | None = None
    memory_context: str | None = None


class KnowledgeSearchOutput(BaseModel):
    answer: str
    sources: list[dict]
    citations: list[dict]
    metadata: dict
