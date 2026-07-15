import json

from backend.app.agents.langgraph.runtime import LangGraphAgentRuntime
from backend.app.agents.state import AgentRuntimeRequest


class TraceGraph:
    def invoke(self, state, config=None):
        state["final_answer"] = "trace answer"
        state["knowledge"] = {
            "answer": "trace answer",
            "sources": [{"source": "doc.pdf", "document_id": 12}],
            "citations": [{"source": "doc.pdf", "document_id": 12}],
            "metadata": {"retriever_mode": "hybrid"},
        }
        state["tool_calls"].append(
            {
                "id": "call_1",
                "tool_call_id": "call_1",
                "tool_name": "knowledge_search",
                "name": "knowledge_search",
                "arguments": {"query": state["query"], "api_key": "secret-value"},
                "status": "success",
                "duration_ms": 3.5,
            }
        )
        state["observations"].append(
            {
                "tool_call_id": "call_1",
                "tool_name": "knowledge_search",
                "success": True,
                "raw_result": {"answer": "trace answer", "token": "secret-token"},
                "metadata": {
                    "provider": "builtin",
                    "cache_hit": False,
                    "retry_count": 0,
                    "duration_ms": 3.5,
                },
            }
        )
        state["metadata"]["checkpoint"] = {
            "enabled": True,
            "provider": "memory",
            "checkpoint_id": "checkpoint-1",
            "saved": True,
        }
        state["metadata"]["memory"].update(
            {"summary_used": False, "token_budget_used": 128}
        )
        state["metadata"]["trace"].extend(
            [
                {
                    "step": "planner_completed",
                    "name": "planner",
                    "node": "planner",
                    "event": "planner_completed",
                    "input": {"query": state["query"]},
                    "output": {"action": "tool_calls"},
                    "input_summary": {"preview": "planner input"},
                    "output_summary": {"preview": "planner output"},
                    "duration_ms": 1.2,
                    "status": "success",
                    "error": None,
                },
                {
                    "step": "tool_completed",
                    "name": "tool",
                    "node": "tool",
                    "event": "tool_completed",
                    "input": {
                        "tool_name": "knowledge_search",
                        "arguments": {"api_key": "secret-value"},
                    },
                    "output": {
                        "success": True,
                        "metadata": {"provider": "builtin", "usage": {"total_tokens": 5}},
                    },
                    "input_summary": {"preview": "tool input"},
                    "output_summary": {"preview": "tool output"},
                    "tool_call_id": "call_1",
                    "tool_name": "knowledge_search",
                    "duration_ms": 3.5,
                    "status": "success",
                    "error": None,
                },
                {
                    "step": "final_answer",
                    "name": "final",
                    "node": "final",
                    "event": "final_answer",
                    "input": {"query": state["query"]},
                    "output": {"answer": "trace answer"},
                    "input_summary": {"preview": "final input"},
                    "output_summary": {"preview": "final output"},
                    "duration_ms": 2.1,
                    "status": "success",
                    "error": None,
                },
            ]
        )
        return state


class BlockedToolGraph:
    def invoke(self, state, config=None):
        state["final_answer"] = "blocked"
        state["tool_calls"].append(
            {
                "id": "call_1",
                "tool_call_id": "call_1",
                "tool_name": "echo",
                "name": "echo",
                "arguments": {"text": "x"},
                "status": "blocked",
                "duration_ms": None,
            }
        )
        state["observations"].append(
            {
                "tool_call_id": "call_1",
                "tool_name": "echo",
                "success": False,
                "error": "tool_not_allowed",
                "metadata": {
                    "status": "blocked",
                    "reason": "tool_not_allowed",
                    "error_type": "tool_permission_error",
                },
            }
        )
        state["metadata"]["tool_scope"].update(
            {
                "allowed_tools": ["knowledge_search"],
                "visible_tools": ["knowledge_search"],
                "selected_tools": ["echo"],
                "blocked_tools": ["echo"],
                "tool_permission_result": "blocked",
                "tool_permission_reason": "tool_not_allowed",
                "tool_scope_source": "agent_definition",
            }
        )
        return state


class ValidationErrorToolGraph:
    def invoke(self, state, config=None):
        state["final_answer"] = "validation failed"
        state["tool_calls"].append(
            {
                "id": "call_1",
                "tool_call_id": "call_1",
                "tool_name": "calculator",
                "name": "calculator",
                "arguments": {"expression": {"bad": "type"}},
                "status": "failed",
                "duration_ms": 0.1,
            }
        )
        state["observations"].append(
            {
                "tool_call_id": "call_1",
                "tool_name": "calculator",
                "success": False,
                "error": "invalid tool arguments",
                "metadata": {
                    "provider": "builtin",
                    "error_type": "tool_validation_error",
                    "duration_ms": 0.1,
                },
            }
        )
        return state


class RuntimeErrorToolGraph:
    def invoke(self, state, config=None):
        state["final_answer"] = "runtime failed"
        state["tool_calls"].append(
            {
                "id": "call_1",
                "tool_call_id": "call_1",
                "tool_name": "calculator",
                "name": "calculator",
                "arguments": {"expression": "1 / 0"},
                "status": "failed",
                "duration_ms": 0.2,
            }
        )
        state["observations"].append(
            {
                "tool_call_id": "call_1",
                "tool_name": "calculator",
                "success": False,
                "error": "division by zero",
                "metadata": {
                    "provider": "builtin",
                    "error_type": "tool_runtime_error",
                    "duration_ms": 0.2,
                },
            }
        )
        return state


def test_agent_trace_result_contains_unified_contract() -> None:
    result = LangGraphAgentRuntime(graph_app=TraceGraph()).run(
        AgentRuntimeRequest(
            query="trace me",
            agent_id="knowledge_research_agent",
            knowledge_base_id=7,
        )
    )

    trace = result.metadata["agent_trace"]

    assert trace["trace_id"] == result.metadata["trace_id"]
    assert trace["runtime"] == "langgraph_v2"
    assert trace["agent_id"] == "knowledge_research_agent"
    assert trace["agent_definition"]["agent_id"] == "knowledge_research_agent"
    assert trace["planner"]["planner_strategy"] == "json_plan"
    assert [node["node"] for node in trace["graph_nodes"]] == ["planner", "tool", "final"]
    assert trace["tool_scope"]["allowed_tools"] == ["knowledge_search", "calculator"]
    assert trace["tool_calls"][0]["tool_name"] == "knowledge_search"
    assert trace["retrieval"]["retrieval_used"] is True
    assert trace["evidence"]["grounded_answer"] is True
    assert trace["memory"]["memory_enabled"] is True
    assert trace["checkpoint"]["checkpoint_enabled"] is True
    assert trace["final_answer"]["answer_length"] == len("trace answer")
    assert trace["timing"]["tool_duration_ms"] == 3.5
    assert trace["token_usage"]["total_tokens"] == 5


def test_agent_trace_redacts_sensitive_values() -> None:
    result = LangGraphAgentRuntime(graph_app=TraceGraph()).run(
        AgentRuntimeRequest(query="trace me", agent_id="knowledge_research_agent")
    )

    raw = json.dumps(result.metadata["agent_trace"], ensure_ascii=False)

    assert "secret-value" not in raw
    assert "secret-token" not in raw
    assert "[REDACTED]" in raw


def test_agent_trace_records_blocked_tool_call() -> None:
    result = LangGraphAgentRuntime(graph_app=BlockedToolGraph()).run(
        AgentRuntimeRequest(query="blocked", agent_id="knowledge_research_agent")
    )

    trace = result.metadata["agent_trace"]

    assert trace["tool_scope"]["permission_result"] == "blocked"
    assert trace["tool_scope"]["permission_reason"] == "tool_not_allowed"
    assert trace["tool_calls"][0]["status"] == "blocked"
    assert trace["tool_calls"][0]["error_type"] == "tool_permission_error"


def test_agent_trace_records_tool_validation_error() -> None:
    result = LangGraphAgentRuntime(graph_app=ValidationErrorToolGraph()).run(
        AgentRuntimeRequest(query="validate", agent_id="general_agent")
    )

    trace = result.metadata["agent_trace"]

    assert trace["tool_calls"][0]["status"] == "failed"
    assert trace["tool_calls"][0]["error_type"] == "tool_validation_error"
    assert trace["tool_calls"][0]["error"] == "invalid tool arguments"


def test_agent_trace_records_tool_runtime_error() -> None:
    result = LangGraphAgentRuntime(graph_app=RuntimeErrorToolGraph()).run(
        AgentRuntimeRequest(query="runtime", agent_id="general_agent")
    )

    trace = result.metadata["agent_trace"]

    assert trace["tool_calls"][0]["status"] == "failed"
    assert trace["tool_calls"][0]["error_type"] == "tool_runtime_error"
    assert trace["tool_calls"][0]["error"] == "division by zero"


def test_agent_trace_fail_open(monkeypatch) -> None:
    def broken_builder(**kwargs):
        raise RuntimeError("trace boom")

    monkeypatch.setattr(
        "backend.app.agents.langgraph.runtime.build_agent_trace_result",
        broken_builder,
    )

    result = LangGraphAgentRuntime(graph_app=TraceGraph()).run(
        AgentRuntimeRequest(query="trace me", agent_id="knowledge_research_agent")
    )

    assert result.answer == "trace answer"
    assert result.metadata["trace_failed"] is True
    assert "trace boom" in result.metadata["trace_error"]
    assert "agent_trace" not in result.metadata


def test_no_second_agent_trace_contract() -> None:
    from pathlib import Path

    names = [
        path.name
        for path in Path("backend/app/agents").rglob("*.py")
        if "trace" in path.name
    ]

    assert names.count("trace.py") == 1
    assert names.count("trace_builder.py") == 1
