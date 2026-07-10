from pydantic import BaseModel, Field


class MemoryState(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    current_plan: dict | None = None
    current_step: str | None = None
    planner_output: dict | None = None
    workflow_state: dict = Field(default_factory=dict)
    trace_id: str | None = None
    session_metadata: dict = Field(default_factory=dict)
