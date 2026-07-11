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
    run_id: str | None = None
    session_id: str | None = None
    node: str | None = None
    event: str | None = None
    input_summary: dict = Field(default_factory=dict)
    output_summary: dict = Field(default_factory=dict)
    tool_call_id: str | None = None
    tool_name: str | None = None
    llm_call_count: int | None = None
    tool_call_count: int | None = None
    reflection_count: int | None = None
    budget_remaining: dict = Field(default_factory=dict)
