from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from backend.app.agents.trace import AgentTraceResult, AgentTraceStep

SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|cookie|password|passwd|secret|token|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
MAX_SUMMARY_CHARS = 800
MAX_STRING_CHARS = 1200
MAX_DEPTH = 8


def build_agent_trace_result(
    *,
    state: dict,
    metadata: dict,
    trace: list[AgentTraceStep],
    answer: str,
    sources: list[dict],
    citations: list[dict],
) -> AgentTraceResult:
    return AgentTraceResult(
        trace_id=_optional_str(metadata.get("trace_id")),
        runtime=str(metadata.get("runtime") or "langgraph_v2"),
        agent_id=_optional_str(metadata.get("agent_id")),
        agent_definition=_agent_definition_trace(metadata),
        planner=_planner_trace(metadata=metadata, trace=trace),
        plan_steps=_plan_steps(state),
        graph_nodes=_graph_nodes(trace),
        tool_scope=_tool_scope_trace(metadata),
        tool_calls=_tool_call_trace(state=state, trace=trace),
        retrieval=_retrieval_trace(metadata),
        evidence=_evidence_trace(metadata=metadata, sources=sources, citations=citations),
        memory=_memory_trace(metadata),
        checkpoint=_checkpoint_trace(metadata),
        reflection=sanitize(metadata.get("reflection", {})),
        final_answer=_final_answer_trace(metadata=metadata, trace=trace, answer=answer),
        errors=_errors(metadata=metadata, trace=trace),
        timing=_timing(metadata=metadata, trace=trace),
        token_usage=_token_usage(trace),
        metadata={
            "schema": "agent_trace_result.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "trace_step_count": len(trace),
        },
    )


def sanitize(value: Any, *, parent_key: str = "", depth: int = 0) -> Any:
    if SENSITIVE_KEY_PATTERN.search(parent_key):
        return "[REDACTED]"
    if depth > MAX_DEPTH:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        return {
            str(key): sanitize(item, parent_key=str(key), depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize(item, parent_key=parent_key, depth=depth + 1)
            for item in value[:50]
        ]
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        return value[:MAX_STRING_CHARS] + "...[truncated]"
    return value


def summary(value: Any) -> dict:
    sanitized = sanitize(value)
    raw = json.dumps(sanitized, ensure_ascii=False, default=str)
    if len(raw) > MAX_SUMMARY_CHARS:
        raw = raw[:MAX_SUMMARY_CHARS] + "...[truncated]"
    return {"preview": raw}


def _agent_definition_trace(metadata: dict) -> dict:
    definition = metadata.get("agent_definition")
    if not isinstance(definition, dict):
        return {}
    return sanitize(
        {
            "agent_id": metadata.get("agent_id") or definition.get("id"),
            "definition_name": definition.get("name"),
            "definition_version": metadata.get("agent_definition_version")
            or definition.get("version"),
            "planner_strategy": metadata.get("planner_strategy")
            or definition.get("planner_strategy"),
            "max_steps": metadata.get("max_steps") or definition.get("max_steps"),
            "timeout_seconds": metadata.get("timeout_seconds")
            or definition.get("timeout_seconds"),
            "output_mode": metadata.get("output_mode") or definition.get("output_mode"),
            "retrieval_policy": metadata.get("retrieval_policy")
            or definition.get("retrieval_policy"),
            "memory_policy": metadata.get("memory_policy") or definition.get("memory_policy"),
            "tool_allowlist": metadata.get("tool_allowlist")
            or definition.get("tool_allowlist"),
            "workflow_allowlist": metadata.get("workflow_allowlist")
            or definition.get("workflow_allowlist"),
            "model_config": {
                "provider": metadata.get("model_provider"),
                "model": metadata.get("model"),
                "public_parameter_keys": metadata.get("model_config_keys", []),
            },
        }
    )


def _planner_trace(*, metadata: dict, trace: list[AgentTraceStep]) -> dict:
    planner_step = next((step for step in trace if step.node == "planner"), None)
    tool_registry = metadata.get("tool_registry", {})
    tool_scope = metadata.get("tool_scope", {})
    return sanitize(
        {
            "planner_strategy": metadata.get("agent_loop", {}).get("planner_strategy")
            or metadata.get("planner_strategy"),
            "planner_input_summary": planner_step.input_summary if planner_step else {},
            "visible_tools": tool_scope.get("visible_tools")
            or tool_registry.get("available_tools", []),
            "visible_workflows": metadata.get("workflow_allowlist", []),
            "generated_plan": planner_step.output_summary if planner_step else {},
            "selected_action": _path(planner_step.output, "action") if planner_step else None,
            "selected_tool": _path(planner_step.output, "tool_calls", 0, "tool_name")
            if planner_step
            else None,
            "planner_status": _normalize_status(planner_step.status) if planner_step else None,
            "duration_ms": planner_step.duration_ms if planner_step else None,
            "error": planner_step.error if planner_step else None,
        }
    )


def _plan_steps(state: dict) -> list[dict]:
    pending = state.get("pending_tool_calls", [])
    tool_calls = state.get("tool_calls", [])
    raw_steps = pending if pending else tool_calls
    if not isinstance(raw_steps, list):
        return []
    return [sanitize(item) for item in raw_steps if isinstance(item, dict)]


def _graph_nodes(trace: list[AgentTraceStep]) -> list[dict]:
    return [
        sanitize(
            {
                "node": step.node or step.name,
                "step": step.step,
                "status": _normalize_status(step.status),
                "input_summary": step.input_summary or summary(step.input),
                "output_summary": step.output_summary or summary(step.output),
                "duration_ms": step.duration_ms,
                "error": step.error,
                "metadata": {
                    "tool_call_id": step.tool_call_id,
                    "tool_name": step.tool_name,
                    "llm_call_count": step.llm_call_count,
                    "tool_call_count": step.tool_call_count,
                    "reflection_count": step.reflection_count,
                },
            }
        )
        for step in trace
    ]


def _tool_scope_trace(metadata: dict) -> dict:
    scope = metadata.get("tool_scope", {})
    if not isinstance(scope, dict):
        return {}
    return sanitize(
        {
            "allowed_tools": scope.get("allowed_tools", metadata.get("tool_allowlist", [])),
            "visible_tools": scope.get("visible_tools", []),
            "selected_tools": scope.get("selected_tools", []),
            "blocked_tools": scope.get("blocked_tools", []),
            "tool_scope_source": scope.get("tool_scope_source", "agent_definition"),
            "permission_result": scope.get("tool_permission_result"),
            "permission_reason": scope.get("tool_permission_reason"),
        }
    )


def _tool_call_trace(*, state: dict, trace: list[AgentTraceStep]) -> list[dict]:
    observations = {
        item.get("tool_call_id"): item
        for item in state.get("observations", [])
        if isinstance(item, dict)
    }
    results = []
    for call in state.get("tool_calls", []):
        if not isinstance(call, dict):
            continue
        observation = observations.get(call.get("tool_call_id") or call.get("id"), {})
        metadata = observation.get("metadata") if isinstance(observation, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        results.append(
            sanitize(
                {
                    "tool_name": call.get("tool_name") or call.get("name"),
                    "provider": metadata.get("provider"),
                    "arguments": call.get("arguments", {}),
                    "allowed": call.get("status") != "blocked",
                    "status": call.get("status"),
                    "result_summary": summary(observation.get("raw_result")),
                    "retry_count": metadata.get("retry_count"),
                    "cache_hit": metadata.get("cache_hit"),
                    "duration_ms": call.get("duration_ms") or metadata.get("duration_ms"),
                    "error_type": metadata.get("error_type"),
                    "error": observation.get("error"),
                }
            )
        )
    if results:
        return results
    return [
        sanitize(
            {
                "tool_name": step.tool_name,
                "provider": _path(step.output, "metadata", "provider"),
                "arguments": _path(step.input, "arguments") or {},
                "allowed": step.status != "blocked",
                "status": _normalize_status(step.status),
                "result_summary": step.output_summary,
                "retry_count": _path(step.output, "metadata", "retry_count"),
                "cache_hit": _path(step.output, "metadata", "cache_hit"),
                "duration_ms": step.duration_ms,
                "error_type": _path(step.output, "metadata", "error_type"),
                "error": step.error,
            }
        )
        for step in trace
        if step.tool_name or step.node == "tool"
    ]


def _retrieval_trace(metadata: dict) -> dict:
    knowledge_metadata = metadata.get("knowledge_metadata", {})
    return sanitize(
        {
            "retrieval_required": metadata.get("retrieval_required"),
            "retrieval_used": metadata.get("retrieval_used"),
            "knowledge_base_id": metadata.get("knowledge_base_id"),
            "selected_document_ids": metadata.get("selected_document_ids", []),
            "retrieval_error": metadata.get("retrieval_error")
            or (knowledge_metadata.get("error") if isinstance(knowledge_metadata, dict) else None),
            "metadata": knowledge_metadata if isinstance(knowledge_metadata, dict) else {},
        }
    )


def _evidence_trace(*, metadata: dict, sources: list[dict], citations: list[dict]) -> dict:
    return sanitize(
        {
            "evidence_count": metadata.get("evidence_count", 0),
            "source_count": metadata.get("source_count", len(sources)),
            "citation_count": metadata.get("citation_count", len(citations)),
            "no_evidence": metadata.get("no_evidence", False),
            "grounded_answer": metadata.get("grounded_answer", False),
            "selected_document_ids": metadata.get("selected_document_ids", []),
        }
    )


def _memory_trace(metadata: dict) -> dict:
    memory = metadata.get("memory", {})
    session = metadata.get("session", {})
    return sanitize(
        {
            "memory_enabled": bool(memory),
            "memory_policy": metadata.get("memory_policy", {}),
            "loaded_message_count": session.get("loaded_message_count"),
            "summary_used": memory.get("summary_used"),
            "token_budget_used": memory.get("token_budget_used"),
            "provider": memory.get("provider"),
            "session_loaded": memory.get("session_loaded") or session.get("loaded"),
            "error": memory.get("error"),
        }
    )


def _checkpoint_trace(metadata: dict) -> dict:
    checkpoint = metadata.get("checkpoint", {})
    session = metadata.get("session", {})
    return sanitize(
        {
            "checkpoint_enabled": checkpoint.get("enabled", False),
            "checkpoint_provider": checkpoint.get("provider"),
            "thread_id": session.get("session_id"),
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "checkpoint_restored": checkpoint.get("restored", session.get("loaded")),
            "checkpoint_saved": checkpoint.get("saved"),
            "error": checkpoint.get("error"),
        }
    )


def _final_answer_trace(
    *,
    metadata: dict,
    trace: list[AgentTraceStep],
    answer: str,
) -> dict:
    final_step = next((step for step in reversed(trace) if step.node == "final"), None)
    return sanitize(
        {
            "status": _normalize_status(final_step.status) if final_step else "completed",
            "answer_length": len(answer),
            "grounded_answer": metadata.get("grounded_answer"),
            "no_evidence": metadata.get("no_evidence"),
            "source_count": metadata.get("source_count"),
            "citation_count": metadata.get("citation_count"),
            "duration_ms": final_step.duration_ms if final_step else None,
        }
    )


def _errors(*, metadata: dict, trace: list[AgentTraceStep]) -> list[dict]:
    errors = []
    for step in trace:
        if step.error:
            errors.append(
                {
                    "stage": step.step,
                    "node": step.node,
                    "error_type": _path(step.output, "metadata", "error_type"),
                    "message": step.error,
                    "recoverable": step.status not in {"failed", "error"},
                    "fallback_used": _path(step.output, "fallback_used"),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
    runtime_error = metadata.get("runtime_error")
    if isinstance(runtime_error, dict):
        errors.append(
            {
                "stage": "runtime",
                "node": None,
                "error_type": runtime_error.get("type"),
                "message": runtime_error.get("message"),
                "recoverable": False,
                "fallback_used": False,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    for item in metadata.get("errors", []):
        errors.append(
            {
                "stage": "metadata",
                "node": None,
                "error_type": None,
                "message": str(item),
                "recoverable": False,
                "fallback_used": False,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    return sanitize(errors)


def _timing(*, metadata: dict, trace: list[AgentTraceStep]) -> dict:
    return {
        "total_duration_ms": _first_number(
            metadata.get("agent_loop", {}).get("duration_ms"),
            metadata.get("async_runtime", {}).get("duration_ms"),
        ),
        "planner_duration_ms": _sum_duration(trace, node="planner"),
        "tool_duration_ms": _sum_duration(trace, node="tool"),
        "retrieval_duration_ms": _retrieval_duration(trace),
        "llm_duration_ms": _sum_duration(trace, node="final"),
        "node_duration_ms": {
            str(step.node or step.name or step.step): step.duration_ms
            for step in trace
        },
    }


def _token_usage(trace: list[AgentTraceStep]) -> dict:
    usage_items = []
    for step in trace:
        usage = _path(step.output, "metadata", "usage") or _path(step.output, "usage")
        if isinstance(usage, dict):
            usage_items.append(usage)
    if not usage_items:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    input_tokens = _sum_usage(usage_items, "input_tokens", "prompt_tokens")
    output_tokens = _sum_usage(usage_items, "output_tokens", "completion_tokens")
    total_tokens = _sum_usage(usage_items, "total_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _normalize_status(status: str | None) -> str:
    if status in {"success", "ok", "passed"}:
        return "completed"
    if status in {"error"}:
        return "failed"
    if status in {"blocked", "skipped", "pending", "running", "failed", "completed"}:
        return status
    return status or "completed"


def _path(value: Any, *path: Any) -> Any:
    current = value
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or len(current) <= key:
                return None
            current = current[key]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
    return current


def _sum_duration(trace: list[AgentTraceStep], *, node: str) -> float | None:
    values = [step.duration_ms for step in trace if step.node == node]
    return round(sum(values), 2) if values else None


def _retrieval_duration(trace: list[AgentTraceStep]) -> float | None:
    values = [
        step.duration_ms
        for step in trace
        if step.tool_name == "knowledge_search"
        or _path(step.input, "tool_name") == "knowledge_search"
    ]
    return round(sum(values), 2) if values else None


def _sum_usage(usage_items: list[dict], *keys: str) -> int | None:
    total = 0
    found = False
    for usage in usage_items:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                total += value
                found = True
                break
    return total if found else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return round(float(value), 2)
    return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
