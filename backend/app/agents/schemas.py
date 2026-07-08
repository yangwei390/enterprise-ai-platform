from backend.app.agents.state import AgentRuntimeResult
from pydantic import BaseModel, Field, field_validator


class AgentChatRequest(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    conversation_id: int | None = None
    memory_context: str | None = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query不能为空")
        return query


class AgentChatResponseData(AgentRuntimeResult):
    pass
