from backend.app.config.settings import settings
from backend.app.workflows.langgraph.schemas import (
    WorkflowDefinitionV2,
    WorkflowEdgeDefinition,
    WorkflowNodeDefinition,
)


def default_agent_workflow_v2() -> WorkflowDefinitionV2:
    return WorkflowDefinitionV2(
        id="default_agent_workflow_v2",
        name="Default Agent Workflow V2",
        description="Start -> Agent -> Condition -> optional Tool -> Final",
        entry_node="start",
        max_steps=settings.WORKFLOW_MAX_STEPS_DEFAULT,
        nodes=[
            WorkflowNodeDefinition(id="start", type="start"),
            WorkflowNodeDefinition(
                id="agent",
                type="agent",
                input_mapping={
                    "query": "{{query}}",
                    "knowledge_base_id": "{{knowledge_base_id}}",
                    "metadata": "{{metadata}}",
                },
            ),
            WorkflowNodeDefinition(
                id="tool_required",
                type="condition",
                config={
                    "condition_key": "agent.tool_calls",
                    "operator": "not_empty",
                    "routes": {
                        "true": "tool",
                        "false": "final",
                    },
                    "default_route": "final",
                },
            ),
            WorkflowNodeDefinition(
                id="tool",
                type="tool",
                config={
                    "tool_name": "echo",
                },
                input_mapping={"text": "{{agent.answer}}"},
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[
            WorkflowEdgeDefinition(source="start", target="agent"),
            WorkflowEdgeDefinition(source="agent", target="tool_required"),
            WorkflowEdgeDefinition(source="tool_required", target="tool", condition="true"),
            WorkflowEdgeDefinition(source="tool_required", target="final", condition="false"),
            WorkflowEdgeDefinition(source="tool", target="final"),
        ],
    )


def approval_knowledge_workflow_v2() -> WorkflowDefinitionV2:
    return WorkflowDefinitionV2(
        id="approval_knowledge_workflow_v2",
        name="Approval Knowledge Workflow V2",
        description="Knowledge tool -> approval -> final",
        entry_node="knowledge",
        max_steps=settings.WORKFLOW_MAX_STEPS_DEFAULT,
        nodes=[
            WorkflowNodeDefinition(
                id="knowledge",
                type="tool",
                config={"tool_name": "knowledge_search", "fail_open": True},
                input_mapping={
                    "query": "{{query}}",
                    "knowledge_base_id": "{{knowledge_base_id}}",
                },
            ),
            WorkflowNodeDefinition(
                id="approval",
                type="approval",
                config={
                    "summary": "请审批知识库回答结果",
                    "routes": {
                        "approved": "final",
                        "rejected": "rejected_final",
                        "modified": "final",
                    },
                },
            ),
            WorkflowNodeDefinition(id="final", type="final"),
            WorkflowNodeDefinition(
                id="rejected_final",
                type="echo",
                input_mapping={"text": "审批已拒绝"},
            ),
        ],
        edges=[
            WorkflowEdgeDefinition(source="knowledge", target="approval"),
            WorkflowEdgeDefinition(source="approval", target="final", condition="approved"),
            WorkflowEdgeDefinition(
                source="approval",
                target="rejected_final",
                condition="rejected",
            ),
            WorkflowEdgeDefinition(source="approval", target="final", condition="modified"),
        ],
    )


def get_workflow_definition_v2(
    workflow_id: str | None,
    definition: WorkflowDefinitionV2 | None = None,
) -> WorkflowDefinitionV2:
    if definition is not None:
        return definition
    if workflow_id in (None, "default_agent_workflow_v2"):
        return default_agent_workflow_v2()
    if workflow_id == "approval_knowledge_workflow_v2":
        return approval_knowledge_workflow_v2()
    raise ValueError(f"Unsupported workflow_id: {workflow_id}")


def list_workflow_definitions_v2() -> list[WorkflowDefinitionV2]:
    return [default_agent_workflow_v2(), approval_knowledge_workflow_v2()]
