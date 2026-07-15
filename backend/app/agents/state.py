from backend.app.agents.trace import AgentTraceStep
from pydantic import BaseModel, Field


class AgentRuntimeRequest(BaseModel):
    query: str
    agent_id: str | None = None
    conversation_id: int | None = None
    knowledge_base_id: int | None = None
    memory_context: str | None = None
    metadata: dict = Field(default_factory=dict)


class PlannerDecision(BaseModel):
    action: str
    tool_name: str | None = None
    reason: str
    confidence: float


class AgentState(BaseModel):
    query: str
    conversation_id: int | None = None
    knowledge_base_id: int | None = None
    memory_context: str | None = None
    planner_decision: PlannerDecision | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)
    final_answer: str | None = None
    metadata: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class AgentRuntimeResult(BaseModel):
    answer: str
    action: str
    tool_calls: list[dict] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    trace: list[AgentTraceStep] = Field(default_factory=list)
