from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowNodeType(StrEnum):
    START = "start"
    END = "end"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    CONDITION = "condition"


class WorkflowNode(BaseModel):
    id: str
    type: str
    name: str | None = None
    config: dict = Field(default_factory=dict)
    next: list[str] = Field(default_factory=list)


class WorkflowDefinition(BaseModel):
    id: str
    name: str
    description: str | None = None
    nodes: list[WorkflowNode]
    start_node_id: str


class WorkflowState(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    node_status: dict[str, str] = Field(default_factory=dict)
    current_node_id: str | None = None
    status: str = WorkflowStatus.PENDING
    error: str | None = None


class WorkflowArtifact(BaseModel):
    key: str
    value: dict | str | int | float | list | None
    metadata: dict = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    workflow_id: str
    status: str
    state: WorkflowState
    artifacts: list[WorkflowArtifact]
    logs: list[dict] = Field(default_factory=list)
    error: str | None = None
