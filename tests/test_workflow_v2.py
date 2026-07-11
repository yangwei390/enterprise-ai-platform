import asyncio
from time import perf_counter
from typing import Any

import pytest
from backend.app.agents import AgentRuntimeResult
from backend.app.api.workflow import router as workflow_router
from backend.app.config.settings import settings
from backend.app.tools import BaseTool, ToolResult
from backend.app.tools.registry import get_tool_registry
from backend.app.workflows.factory import WorkflowRuntimeFactory
from backend.app.workflows.langgraph.checkpoint import WorkflowCheckpointAdapter
from backend.app.workflows.langgraph.errors import (
    WorkflowAlreadyCompletedError,
    WorkflowResumeError,
    WorkflowValidationError,
)
from backend.app.workflows.langgraph.runtime import LangGraphWorkflowRuntime
from backend.app.workflows.langgraph.schemas import (
    WorkflowDefinitionV2,
    WorkflowEdgeDefinition,
    WorkflowNodeDefinition,
    WorkflowResumeCommand,
    WorkflowResumeRequest,
    WorkflowRunRequestV2,
)
from backend.app.workflows.langgraph.validator import WorkflowDefinitionValidator
from backend.app.workflows.v1 import WorkflowRuntimeV1
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class WorkflowToolArgs(BaseModel):
    text: str | None = None
    delay: float | None = None
    value: int | None = None


class WorkflowEchoTestTool(BaseTool):
    name = "workflow_v2_echo_test"
    description = "workflow v2 echo test"
    args_schema = WorkflowToolArgs
    source = "test"

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(
            name=self.name,
            success=True,
            result={"answer": arguments.get("text") or "echo ok"},
        )

    async def arun(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(float(arguments.get("delay") or 0))
        return self.run(arguments)


class WorkflowFailTestTool(BaseTool):
    name = "workflow_v2_fail_test"
    description = "workflow v2 fail test"
    args_schema = WorkflowToolArgs
    source = "test"

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=False, error="tool failed")

    async def arun(self, arguments: dict) -> ToolResult:
        return self.run(arguments)


class WorkflowMCPTestTool(BaseTool):
    name = "mcp__workflow_v2__echo"
    description = "fake mcp tool through registry"
    args_schema = WorkflowToolArgs
    source = "mcp"

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(
            name=self.name,
            success=True,
            result={"answer": f"mcp:{arguments.get('text')}"},
        )

    async def arun(self, arguments: dict) -> ToolResult:
        return self.run(arguments)


class WorkflowSleepTool(BaseTool):
    name = "workflow_v2_sleep_test"
    description = "workflow v2 sleep test"
    args_schema = WorkflowToolArgs
    source = "test"

    active = 0
    max_active = 0

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(name=self.name, success=True, result={"answer": "slept"})

    async def arun(self, arguments: dict) -> ToolResult:
        WorkflowSleepTool.active += 1
        WorkflowSleepTool.max_active = max(WorkflowSleepTool.max_active, WorkflowSleepTool.active)
        await asyncio.sleep(float(arguments.get("delay") or 0.05))
        WorkflowSleepTool.active -= 1
        return self.run(arguments)


class FakeAgentRuntime:
    called = False

    async def arun(self, request):
        FakeAgentRuntime.called = True
        return AgentRuntimeResult(
            answer=f"agent:{request.query}",
            action="direct_answer",
            tool_calls=[],
            observations=[],
            sources=[],
            citations=[],
            metadata={"fake": True},
            trace=[],
        )


def register_test_tool(tool: BaseTool) -> None:
    registry = get_tool_registry()
    if registry.contains(tool.name):
        registry.unregister(tool.name)
    registry.register(tool)


def unregister_test_tools() -> None:
    registry = get_tool_registry()
    for name in [
        "workflow_v2_echo_test",
        "workflow_v2_fail_test",
        "mcp__workflow_v2__echo",
        "workflow_v2_sleep_test",
    ]:
        if registry.contains(name):
            registry.unregister(name)


def simple_echo_definition() -> WorkflowDefinitionV2:
    return WorkflowDefinitionV2(
        id="test_echo_v2",
        name="test echo",
        entry_node="start",
        nodes=[
            WorkflowNodeDefinition(id="start", type="start"),
            WorkflowNodeDefinition(
                id="tool",
                type="tool",
                config={"tool_name": "workflow_v2_echo_test"},
                input_mapping={"text": "{{query}}"},
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[
            WorkflowEdgeDefinition(source="start", target="tool"),
            WorkflowEdgeDefinition(source="tool", target="final"),
        ],
    )


def run_v2(definition: WorkflowDefinitionV2, query: str = "hello"):
    return asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query=query, definition=definition)
        )
    )


def test_workflow_factory_returns_v1(monkeypatch):
    monkeypatch.setattr(settings, "WORKFLOW_RUNTIME", "v1")
    WorkflowRuntimeFactory.reset()

    runtime = WorkflowRuntimeFactory.get_runtime()

    assert isinstance(runtime, WorkflowRuntimeV1)


def test_workflow_factory_returns_langgraph_v2(monkeypatch):
    monkeypatch.setattr(settings, "WORKFLOW_RUNTIME", "langgraph")
    monkeypatch.setattr(settings, "WORKFLOW_V2_ENABLED", True)
    WorkflowRuntimeFactory.reset()

    runtime = WorkflowRuntimeFactory.get_runtime()

    assert isinstance(runtime, LangGraphWorkflowRuntime)


def test_workflow_definition_validation():
    register_test_tool(WorkflowEchoTestTool())

    WorkflowDefinitionValidator().validate(simple_echo_definition())

    unregister_test_tools()


def test_workflow_rejects_duplicate_node_ids():
    definition = WorkflowDefinitionV2(
        id="bad_duplicate",
        name="bad",
        entry_node="a",
        nodes=[
            WorkflowNodeDefinition(id="a", type="start"),
            WorkflowNodeDefinition(id="a", type="final"),
        ],
    )

    with pytest.raises(WorkflowValidationError):
        WorkflowDefinitionValidator().validate(definition)


def test_workflow_rejects_missing_edge_target():
    definition = WorkflowDefinitionV2(
        id="bad_target",
        name="bad",
        entry_node="a",
        nodes=[
            WorkflowNodeDefinition(id="a", type="start"),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[WorkflowEdgeDefinition(source="a", target="missing")],
    )

    with pytest.raises(WorkflowValidationError):
        WorkflowDefinitionValidator().validate(definition)


def test_workflow_rejects_unknown_node_type():
    definition = WorkflowDefinitionV2(
        id="bad_type",
        name="bad",
        entry_node="a",
        nodes=[
            WorkflowNodeDefinition(id="a", type="start"),
            WorkflowNodeDefinition(id="x", type="unknown"),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[
            WorkflowEdgeDefinition(source="a", target="x"),
            WorkflowEdgeDefinition(source="x", target="final"),
        ],
    )

    with pytest.raises(WorkflowValidationError):
        WorkflowDefinitionValidator().validate(definition)


def test_tool_node_uses_tool_executor():
    register_test_tool(WorkflowEchoTestTool())

    result = run_v2(simple_echo_definition(), query="tool answer")

    assert result.status == "completed"
    assert result.answer == "tool answer"
    assert result.node_outputs["tool"]["_tool_result"]["name"] == "workflow_v2_echo_test"
    unregister_test_tools()


def test_agent_node_uses_agent_runtime_factory(monkeypatch):
    FakeAgentRuntime.called = False
    monkeypatch.setattr(
        "backend.app.agents.factory.AgentRuntimeFactory.get_runtime",
        lambda: FakeAgentRuntime(),
    )
    definition = WorkflowDefinitionV2(
        id="test_agent_node",
        name="agent",
        entry_node="agent",
        nodes=[
            WorkflowNodeDefinition(id="agent", type="agent"),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[WorkflowEdgeDefinition(source="agent", target="final")],
    )

    result = run_v2(definition, query="agent query")

    assert FakeAgentRuntime.called is True
    assert result.answer == "agent:agent query"


def test_mcp_tool_runs_through_tool_executor():
    register_test_tool(WorkflowMCPTestTool())
    definition = WorkflowDefinitionV2(
        id="test_mcp_tool",
        name="mcp",
        entry_node="tool",
        nodes=[
            WorkflowNodeDefinition(
                id="tool",
                type="tool",
                config={"tool_name": "mcp__workflow_v2__echo"},
                input_mapping={"text": "hello"},
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[WorkflowEdgeDefinition(source="tool", target="final")],
    )

    result = run_v2(definition)

    assert result.answer == "mcp:hello"
    assert result.node_outputs["tool"]["_tool_result"]["metadata"]["provider"] == "mcp"
    unregister_test_tools()


def condition_definition(operator: str, value: Any = None) -> WorkflowDefinitionV2:
    return WorkflowDefinitionV2(
        id=f"test_condition_{operator}",
        name="condition",
        entry_node="condition",
        nodes=[
            WorkflowNodeDefinition(
                id="condition",
                type="condition",
                config={
                    "condition_key": "flag",
                    "operator": operator,
                    "value": value,
                    "routes": {"true": "true_echo", "false": "false_echo"},
                },
            ),
            WorkflowNodeDefinition(
                id="true_echo",
                type="echo",
                input_mapping={"text": "true branch"},
            ),
            WorkflowNodeDefinition(
                id="false_echo",
                type="echo",
                input_mapping={"text": "false branch"},
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[
            WorkflowEdgeDefinition(source="condition", target="true_echo", condition="true"),
            WorkflowEdgeDefinition(source="condition", target="false_echo", condition="false"),
            WorkflowEdgeDefinition(source="true_echo", target="final"),
            WorkflowEdgeDefinition(source="false_echo", target="final"),
        ],
    )


def test_condition_node_routes_true_branch():
    definition = condition_definition("truthy")

    result = asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query="q", inputs={"flag": True}, definition=definition)
        )
    )

    assert result.answer == "true branch"


def test_condition_node_routes_false_branch():
    definition = condition_definition("truthy")

    result = asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query="q", inputs={"flag": False}, definition=definition)
        )
    )

    assert result.answer == "false branch"


def test_unknown_condition_route_fails():
    definition = WorkflowDefinitionV2(
        id="bad_route_runtime",
        name="bad route",
        entry_node="condition",
        nodes=[
            WorkflowNodeDefinition(
                id="condition",
                type="condition",
                config={
                    "condition_key": "missing",
                    "operator": "exists",
                    "routes": {"true": "final"},
                },
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[WorkflowEdgeDefinition(source="condition", target="final", condition="true")],
    )

    result = run_v2(definition)

    assert result.status == "failed"
    assert "unknown condition route" in result.metadata["workflow_runtime"]["error"]


def test_loop_stops_at_max_steps():
    definition = WorkflowDefinitionV2(
        id="loop_test",
        name="loop",
        entry_node="condition",
        max_steps=3,
        nodes=[
            WorkflowNodeDefinition(
                id="condition",
                type="condition",
                config={
                    "condition_key": "flag",
                    "operator": "truthy",
                    "routes": {"true": "condition", "false": "finish"},
                },
            ),
            WorkflowNodeDefinition(
                id="finish",
                type="echo",
                input_mapping={"text": "done"},
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[
            WorkflowEdgeDefinition(source="condition", target="condition", condition="true"),
            WorkflowEdgeDefinition(source="condition", target="finish", condition="false"),
            WorkflowEdgeDefinition(source="finish", target="final"),
        ],
    )

    result = asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query="q", inputs={"flag": True}, definition=definition)
        )
    )

    assert result.status == "failed"
    assert "max_steps" in result.metadata["workflow_runtime"]["error"]


def parallel_definition(fail_fast: bool = False, fail_branch: bool = False) -> WorkflowDefinitionV2:
    branches = [
        {
            "id": "a",
            "tool_name": "workflow_v2_sleep_test",
            "input": {"text": "a", "delay": 0.05},
        },
        {
            "id": "b",
            "tool_name": "workflow_v2_sleep_test",
            "input": {"text": "b", "delay": 0.05},
        },
    ]
    if fail_branch:
        branches.append({"id": "bad", "tool_name": "workflow_v2_fail_test", "input": {}})
    return WorkflowDefinitionV2(
        id=f"parallel_{fail_fast}_{fail_branch}",
        name="parallel",
        entry_node="parallel",
        nodes=[
            WorkflowNodeDefinition(
                id="parallel",
                type="parallel",
                config={"branches": branches, "fail_fast": fail_fast},
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[WorkflowEdgeDefinition(source="parallel", target="final")],
    )


def test_parallel_branches_execute_concurrently():
    register_test_tool(WorkflowSleepTool())
    WorkflowSleepTool.active = 0
    WorkflowSleepTool.max_active = 0
    started_at = perf_counter()

    result = run_v2(parallel_definition())

    assert result.status == "completed"
    assert WorkflowSleepTool.max_active == 2
    assert (perf_counter() - started_at) < 0.1
    unregister_test_tools()


def test_parallel_branch_collects_errors():
    register_test_tool(WorkflowSleepTool())
    register_test_tool(WorkflowFailTestTool())

    result = run_v2(parallel_definition(fail_fast=False, fail_branch=True))

    assert result.status == "completed"
    assert result.node_outputs["parallel"]["branch_results"]["bad"]["success"] is False
    unregister_test_tools()


def test_parallel_fail_fast():
    register_test_tool(WorkflowSleepTool())
    register_test_tool(WorkflowFailTestTool())

    result = run_v2(parallel_definition(fail_fast=True, fail_branch=True))

    assert result.status == "failed"
    assert "tool failed" in result.metadata["workflow_runtime"]["error"]
    unregister_test_tools()


def test_checkpoint_saved_after_graph_step():
    register_test_tool(WorkflowEchoTestTool())
    runtime = LangGraphWorkflowRuntime()

    result = asyncio.run(
        runtime.arun(WorkflowRunRequestV2(query="checkpoint", definition=simple_echo_definition()))
    )

    assert runtime.get_state(result.thread_id) is not None
    assert "checkpoint_id" in result.metadata["workflow_runtime"]
    unregister_test_tools()


def test_checkpoint_loaded_by_thread_id():
    adapter = WorkflowCheckpointAdapter()
    adapter.save_state("thread-x", {"status": "interrupted", "run_id": "run-x"})

    state = adapter.load_state("thread-x")

    assert state["run_id"] == "run-x"


def approval_definition() -> WorkflowDefinitionV2:
    return WorkflowDefinitionV2(
        id="approval_test",
        name="approval",
        entry_node="approval",
        nodes=[
            WorkflowNodeDefinition(
                id="approval",
                type="approval",
                config={
                    "summary": "approve it",
                    "routes": {
                        "approved": "final",
                        "rejected": "rejected_echo",
                        "modified": "final",
                    },
                },
            ),
            WorkflowNodeDefinition(id="final", type="final"),
            WorkflowNodeDefinition(
                id="rejected_echo",
                type="echo",
                input_mapping={"text": "rejected"},
            ),
        ],
        edges=[
            WorkflowEdgeDefinition(source="approval", target="final", condition="approved"),
            WorkflowEdgeDefinition(source="approval", target="rejected_echo", condition="rejected"),
            WorkflowEdgeDefinition(source="approval", target="final", condition="modified"),
            WorkflowEdgeDefinition(source="rejected_echo", target="final"),
        ],
    )


def test_workflow_interrupt_for_approval():
    result = asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query="approve", definition=approval_definition())
        )
    )

    assert result.status == "interrupted"
    assert result.interrupt["node_id"] == "approval"


def test_workflow_resume_after_approve():
    runtime = LangGraphWorkflowRuntime()
    first = asyncio.run(
        runtime.arun(WorkflowRunRequestV2(query="approve", definition=approval_definition()))
    )

    result = asyncio.run(
        runtime.aresume(
            WorkflowResumeRequest(
                workflow_id="approval_test",
                thread_id=first.thread_id,
                run_id=first.run_id,
                command=WorkflowResumeCommand(action="approve", value={"approved": True}),
            )
        )
    )

    assert result.status == "completed"
    assert [item["node_id"] for item in result.trace][-2:] == ["approval", "final"]


def test_workflow_resume_after_reject():
    runtime = LangGraphWorkflowRuntime()
    first = asyncio.run(
        runtime.arun(WorkflowRunRequestV2(query="approve", definition=approval_definition()))
    )

    result = asyncio.run(
        runtime.aresume(
            WorkflowResumeRequest(
                workflow_id="approval_test",
                thread_id=first.thread_id,
                run_id=first.run_id,
                command=WorkflowResumeCommand(action="reject", value={}),
            )
        )
    )

    assert result.status == "completed"
    assert result.answer == "rejected"


def test_workflow_modify_then_resume():
    runtime = LangGraphWorkflowRuntime()
    first = asyncio.run(
        runtime.arun(WorkflowRunRequestV2(query="approve", definition=approval_definition()))
    )

    result = asyncio.run(
        runtime.aresume(
            WorkflowResumeRequest(
                workflow_id="approval_test",
                thread_id=first.thread_id,
                run_id=first.run_id,
                command=WorkflowResumeCommand(action="modify", value={"answer": "changed"}),
            )
        )
    )

    assert result.status == "completed"
    assert result.metadata["workflow_runtime"]["resumed"] is True


def test_resume_requires_existing_thread():
    with pytest.raises(WorkflowResumeError):
        asyncio.run(
            LangGraphWorkflowRuntime().aresume(
                WorkflowResumeRequest(
                    workflow_id="approval_knowledge_workflow_v2",
                    thread_id="missing",
                    run_id="missing",
                    command=WorkflowResumeCommand(action="approve"),
                )
            )
        )


def test_completed_workflow_cannot_resume():
    register_test_tool(WorkflowEchoTestTool())
    runtime = LangGraphWorkflowRuntime()
    completed = asyncio.run(
        runtime.arun(WorkflowRunRequestV2(query="done", definition=simple_echo_definition()))
    )

    with pytest.raises(WorkflowAlreadyCompletedError):
        asyncio.run(
            runtime.aresume(
                WorkflowResumeRequest(
                    workflow_id="test_echo_v2",
                    thread_id=completed.thread_id,
                    run_id=completed.run_id,
                    command=WorkflowResumeCommand(action="approve"),
                )
            )
        )
    unregister_test_tools()


def test_approval_permission_denied(monkeypatch):
    monkeypatch.setattr(settings, "WORKFLOW_APPROVAL_PERMISSION_ENFORCEMENT", True)
    definition = WorkflowDefinitionV2(
        id="approval_permission",
        name="approval permission",
        entry_node="approval",
        nodes=[
            WorkflowNodeDefinition(
                id="approval",
                type="approval",
                config={
                    "required_permissions": ["workflow.approve"],
                    "routes": {"approved": "final", "rejected": "final"},
                },
            ),
            WorkflowNodeDefinition(id="final", type="final"),
        ],
        edges=[WorkflowEdgeDefinition(source="approval", target="final", condition="approved")],
    )

    result = asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query="approve", definition=definition)
        )
    )

    assert result.status == "failed"
    assert "missing workflow approval permissions" in result.metadata["workflow_runtime"]["error"]


def test_workflow_trace_contains_nodes():
    register_test_tool(WorkflowEchoTestTool())

    result = run_v2(simple_echo_definition())

    assert [item["node_id"] for item in result.trace] == ["start", "tool", "final"]
    assert all("duration_ms" in item for item in result.trace)
    unregister_test_tools()


def test_workflow_metadata_contains_checkpoint():
    register_test_tool(WorkflowEchoTestTool())

    result = run_v2(simple_echo_definition())

    metadata = result.metadata["workflow_runtime"]
    assert metadata["checkpoint_enabled"] is True
    assert metadata["checkpoint_provider"] in {"redis", "memory"}
    unregister_test_tools()


def test_workflow_timeout():
    register_test_tool(WorkflowSleepTool())
    definition = parallel_definition()
    definition.timeout_seconds = 0

    result = run_v2(definition)

    assert result.status == "failed"
    unregister_test_tools()


def test_workflow_cancellation_propagates():
    register_test_tool(WorkflowSleepTool())
    definition = parallel_definition()

    async def run_and_cancel():
        task = asyncio.create_task(
            LangGraphWorkflowRuntime().arun(
                WorkflowRunRequestV2(query="cancel", definition=definition)
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_and_cancel())
    unregister_test_tools()


def test_workflow_astream_emits_events():
    register_test_tool(WorkflowEchoTestTool())

    async def collect():
        events = []
        async for event in LangGraphWorkflowRuntime().astream(
            WorkflowRunRequestV2(query="stream", definition=simple_echo_definition())
        ):
            events.append(event["event"])
        return events

    events = asyncio.run(collect())

    assert events[0] == "workflow_started"
    assert events[-1] == "workflow_completed"
    unregister_test_tools()


def test_workflow_run_api_v1_compatible(monkeypatch):
    class FakeV1Runtime:
        def run(self, request):
            return {"answer": "v1 ok", "node_outputs": {}, "trace": [], "metadata": {}}

    monkeypatch.setattr(
        "backend.app.api.workflow.WorkflowRuntimeFactory.get_runtime",
        lambda provider=None: FakeV1Runtime(),
    )
    app = FastAPI()
    app.include_router(workflow_router)
    client = TestClient(app)

    response = client.post("/workflow/run", json={"query": "hello"})

    assert response.status_code == 200
    assert response.json()["data"]["answer"] == "v1 ok"


def test_workflow_resume_api(monkeypatch):
    class FakeV2Runtime:
        async def aresume(self, request):
            return {"status": "completed", "answer": "resumed"}

    monkeypatch.setattr(
        "backend.app.api.workflow.WorkflowRuntimeFactory.get_runtime",
        lambda provider=None: FakeV2Runtime(),
    )
    app = FastAPI()
    app.include_router(workflow_router)
    client = TestClient(app)

    response = client.post(
        "/workflow/resume",
        json={
            "workflow_id": "approval_knowledge_workflow_v2",
            "thread_id": "thread",
            "run_id": "run",
            "command": {"action": "approve", "value": {}},
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "completed"


def test_debug_workflow_api():
    from backend.app.api.debug import router as debug_router

    app = FastAPI()
    app.include_router(debug_router)
    client = TestClient(app)

    response = client.get("/debug/workflows")

    assert response.status_code == 200
    assert response.json()["data"]["runtime"] in {"v1", "langgraph"}


def test_v1_workflow_still_works():
    from backend.app.workflows.v1 import WorkflowRunRequest, WorkflowRuntimeV1

    result = WorkflowRuntimeV1().run(
        WorkflowRunRequest(
            query="hello",
            definition={
                "id": "v1_echo",
                "name": "v1 echo",
                "nodes": [
                    {"id": "final", "type": "echo", "input": {"text": "{{query}}"}},
                ],
            },
        )
    )

    assert result.answer == "hello"


def test_real_langgraph_workflow_smoke():
    register_test_tool(WorkflowEchoTestTool())

    result = run_v2(simple_echo_definition(), query="real smoke")

    assert result.status == "completed"
    assert result.answer == "real smoke"
    assert result.metadata["workflow_runtime"]["runtime"] == "langgraph"
    unregister_test_tools()


def test_real_checkpoint_resume_smoke():
    first = asyncio.run(
        LangGraphWorkflowRuntime().arun(
            WorkflowRunRequestV2(query="approve", definition=approval_definition())
        )
    )

    assert first.status == "interrupted"
    assert WorkflowCheckpointAdapter().load_state(first.thread_id) is not None


def test_real_human_approval_smoke():
    runtime = LangGraphWorkflowRuntime()
    first = asyncio.run(
        runtime.arun(WorkflowRunRequestV2(query="approve", definition=approval_definition()))
    )
    second = asyncio.run(
        runtime.aresume(
            WorkflowResumeRequest(
                workflow_id="approval_test",
                thread_id=first.thread_id,
                run_id=first.run_id,
                command=WorkflowResumeCommand(action="approve", value={"approved": True}),
            )
        )
    )

    assert first.status == "interrupted"
    assert second.status == "completed"
    assert second.metadata["workflow_runtime"]["resumed"] is True
