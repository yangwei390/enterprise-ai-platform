from pydantic import BaseModel, Field


class AgentTraceStep(BaseModel):
    step: str
    name: str
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    duration_ms: float = 0
    async_execution: bool | None = None
    task_id: str | None = None
    timeout: bool | None = None
    cancelled: bool | None = None
    retry_count: int | None = None
    sync_fallback: bool | None = None
    status: str = "success"
    error: str | None = None
