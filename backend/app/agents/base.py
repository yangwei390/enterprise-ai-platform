from enum import StrEnum

from backend.app.agents.definition import AgentDefinition as AgentDefinition
from backend.app.workflows import WorkflowArtifact
from pydantic import BaseModel, Field


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class AgentStepType(StrEnum):
    PLAN = "plan"
    WORKFLOW = "workflow"
    TOOL = "tool"
    RETRIEVER = "retriever"
    LLM = "llm"
    REFLECTION = "reflection"
    FINAL = "final"


class AgentStep(BaseModel):
    index: int
    type: str
    name: str | None = None
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    status: str = AgentRunStatus.PENDING
    error: str | None = None


class AgentRunRequest(BaseModel):
    task: str
    agent_id: str | None = None
    knowledge_base_id: int | None = None
    enable_tools: bool = True
    enable_memory: bool = True
    conversation_id: int | None = None
    metadata: dict = Field(default_factory=dict)


class AgentRunResult(BaseModel):
    task: str
    status: str
    answer: str | None = None
    steps: list[AgentStep]
    artifacts: list[WorkflowArtifact]
    metadata: dict = Field(default_factory=dict)
    error: str | None = None
