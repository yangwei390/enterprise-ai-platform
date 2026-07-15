import asyncio

import pytest
from backend.app.agents.catalog import AgentCatalog
from backend.app.agents.definition import (
    AgentDefinition,
    AgentDefinitionDisabledError,
    AgentDefinitionNotFoundError,
    AgentDefinitionRegistry,
    get_agent_definition_registry,
    reset_agent_definition_registry,
)
from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.state import AgentRuntimeRequest


@pytest.fixture(autouse=True)
def reset_registry():
    reset_agent_definition_registry()
    yield
    reset_agent_definition_registry()


def _definition(**overrides) -> AgentDefinition:
    data = {
        "id": "custom_agent",
        "name": "Custom Agent",
        "description": "Custom test agent",
        "instructions": "Custom system instructions.",
        "planner_strategy": "json_plan",
        "tool_allowlist": ["knowledge_search"],
        "workflow_allowlist": ["default_agent_workflow_v2"],
        "default_knowledge_base_id": 99,
        "memory_policy": {"enabled": True},
        "retrieval_policy": {"enabled": True},
        "model_config": {"model": "test-model", "temperature": 0.1},
        "max_steps": 3,
        "timeout_seconds": 7,
        "output_mode": "grounded_answer",
        "safety_policy": {"strict": True},
        "enabled": True,
        "version": "test-1",
        "metadata": {"source": "test"},
    }
    data.update(overrides)
    return AgentDefinition(**data)


def test_registry_register_get_and_list() -> None:
    registry = AgentDefinitionRegistry()
    definition = _definition()

    registry.register(definition)

    assert registry.get("custom_agent").id == "custom_agent"
    assert [item.id for item in registry.list()] == ["custom_agent"]


def test_registry_missing_agent_id_returns_clear_error() -> None:
    registry = AgentDefinitionRegistry()

    try:
        registry.get("missing_agent")
    except AgentDefinitionNotFoundError as exc:
        assert "missing_agent" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected AgentDefinitionNotFoundError")


def test_registry_disabled_agent_is_not_executable() -> None:
    registry = AgentDefinitionRegistry(definitions=[_definition(enabled=False)])

    try:
        registry.get("custom_agent")
    except AgentDefinitionDisabledError as exc:
        assert "custom_agent" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected AgentDefinitionDisabledError")


def test_builtin_general_and_knowledge_agents_are_different() -> None:
    registry = reset_agent_definition_registry()
    general = registry.get("general_agent")
    knowledge = registry.get("knowledge_agent")

    assert general.instructions != knowledge.instructions
    assert general.planner_strategy != knowledge.planner_strategy
    assert general.tool_allowlist != knowledge.tool_allowlist
    assert general.retrieval_policy != knowledge.retrieval_policy


def test_runtime_loads_agent_definition_and_applies_policies() -> None:
    registry = reset_agent_definition_registry()
    registry.register(_definition())
    captured = {}

    class CaptureGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            captured["config"] = config
            state["final_answer"] = "ok"
            state["metadata"]["trace"].append(
                {
                    "step": "final_answer",
                    "name": "final_answer",
                    "input": {"query": state["query"]},
                    "output": {"answer": "ok"},
                    "duration_ms": 0,
                    "status": "success",
                    "error": None,
                }
            )
            return state

    result = LangGraphAgentRuntime(graph_app=CaptureGraph()).run(
        AgentRuntimeRequest(query="hello", agent_id="custom_agent")
    )

    state = captured["state"]
    assert state["messages"][0]["role"] == "system"
    assert state["messages"][0]["content"] == "Custom system instructions."
    assert state["knowledge_base_id"] == 99
    assert state["budget"]["max_steps"] == 3
    assert captured["config"]["recursion_limit"] == 11
    assert result.metadata["agent_id"] == "custom_agent"
    assert result.metadata["agent_definition_version"] == "test-1"
    assert result.metadata["planner_strategy"] == "json_plan"
    assert result.metadata["tool_allowlist"] == ["knowledge_search"]
    assert result.metadata["workflow_allowlist"] == ["default_agent_workflow_v2"]
    assert result.metadata["memory_policy"] == {"enabled": True}
    assert result.metadata["retrieval_policy"] == {"enabled": True}
    assert result.metadata["model_config_keys"] == ["model", "temperature"]
    assert "model_config" not in result.metadata["agent_definition"]


def test_runtime_rejects_disabled_agent_without_graph_execution() -> None:
    registry = reset_agent_definition_registry()
    registry.register(_definition(id="disabled_agent", enabled=False))

    class FailingGraph:
        def invoke(self, state, config=None):  # pragma: no cover - must not execute
            raise AssertionError("graph should not execute")

    result = LangGraphAgentRuntime(graph_app=FailingGraph()).run(
        AgentRuntimeRequest(query="hello", agent_id="disabled_agent")
    )

    assert result.action == "failed"
    assert result.metadata["runtime_error"]["type"] == "agent_definition_error"


def test_runtime_uses_definition_timeout_for_async_execution() -> None:
    registry = reset_agent_definition_registry()
    registry.register(_definition(id="async_agent", timeout_seconds=7))

    class AsyncCaptureGraph:
        async def ainvoke(self, state, config=None):
            state["final_answer"] = "async ok"
            return state

    result = asyncio.run(
        LangGraphAgentRuntime(graph_app=AsyncCaptureGraph()).arun(
            AgentRuntimeRequest(query="hello", agent_id="async_agent")
        )
    )

    assert result.answer == "async ok"
    assert result.metadata["async_runtime"]["timeout_seconds"] == 7


def test_catalog_and_runtime_use_same_registry() -> None:
    registry = reset_agent_definition_registry()
    registry_ids = [definition.id for definition in registry.list()]
    catalog_ids = [assistant.id for assistant in AgentCatalog().list_assistants()]

    assert catalog_ids == registry_ids
    assert AgentCatalog().get_assistant("knowledge_agent") is not None
    assert get_agent_definition_registry().get("knowledge_agent").id == "knowledge_agent"
