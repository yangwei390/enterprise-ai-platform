from pydantic import BaseModel, Field


class AgentTraceStep(BaseModel):
    step: str
    name: str
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    duration_ms: float = 0
    status: str = "success"
    error: str | None = None
