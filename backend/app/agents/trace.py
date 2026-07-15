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


class AgentTraceResult(BaseModel):
    trace_id: str | None = None
    runtime: str = "langgraph_v2"
    agent_id: str | None = None
    agent_definition: dict = Field(default_factory=dict)
    planner: dict = Field(default_factory=dict)
    plan_steps: list[dict] = Field(default_factory=list)
    graph_nodes: list[dict] = Field(default_factory=list)
    tool_scope: dict = Field(default_factory=dict)
    tool_calls: list[dict] = Field(default_factory=list)
    retrieval: dict = Field(default_factory=dict)
    evidence: dict = Field(default_factory=dict)
    memory: dict = Field(default_factory=dict)
    checkpoint: dict = Field(default_factory=dict)
    reflection: dict = Field(default_factory=dict)
    final_answer: dict = Field(default_factory=dict)
    errors: list[dict] = Field(default_factory=list)
    timing: dict = Field(default_factory=dict)
    token_usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
