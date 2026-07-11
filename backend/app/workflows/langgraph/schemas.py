from typing import Any

from pydantic import BaseModel, Field, field_validator


class WorkflowNodeDefinition(BaseModel):
    id: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)
    input_mapping: dict[str, Any] = Field(default_factory=dict)
    output_mapping: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    timeout_seconds: float | None = None
    cache_enabled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdgeDefinition(BaseModel):
    source: str
    target: str
    condition: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinitionV2(BaseModel):
    id: str
    name: str
    version: str = "2.0"
    description: str | None = None
    entry_node: str
    nodes: list[WorkflowNodeDefinition]
    edges: list[WorkflowEdgeDefinition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 20
    timeout_seconds: int = 120
    checkpoint_enabled: bool = True
    approval_enabled: bool = True

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_steps must be greater than 0")
        return value


class WorkflowRunRequestV2(BaseModel):
    workflow_id: str | None = "default_agent_workflow_v2"
    query: str
    knowledge_base_id: int | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    definition: WorkflowDefinitionV2 | None = None


class WorkflowResumeCommand(BaseModel):
    action: str
    value: dict[str, Any] = Field(default_factory=dict)


class WorkflowResumeRequest(BaseModel):
    workflow_id: str = "approval_knowledge_workflow_v2"
    thread_id: str
    run_id: str | None = None
    command: WorkflowResumeCommand


class WorkflowApprovalRequest(BaseModel):
    workflow_id: str
    run_id: str
    thread_id: str
    node_id: str
    action: str = "approval_required"
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    required_permissions: list[str] = Field(default_factory=list)
    expires_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunResultV2(BaseModel):
    workflow_id: str
    run_id: str
    thread_id: str
    status: str
    answer: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    node_outputs: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    interrupt: dict[str, Any] | None = None
