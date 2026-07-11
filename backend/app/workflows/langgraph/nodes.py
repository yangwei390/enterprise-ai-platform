import asyncio
import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, cast

from backend.app.agents.state import AgentRuntimeRequest
from backend.app.config.settings import settings
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.tools import ToolCall, ToolExecutor
from backend.app.workflows.langgraph.approval import (
    build_approval_request,
    check_approval_permission,
)
from backend.app.workflows.langgraph.checkpoint import WorkflowCheckpointAdapter
from backend.app.workflows.langgraph.errors import (
    WorkflowConditionError,
    WorkflowMaxStepsExceeded,
    WorkflowNodeError,
)
from backend.app.workflows.langgraph.schemas import (
    WorkflowDefinitionV2,
    WorkflowNodeDefinition,
)
from backend.app.workflows.langgraph.trace import make_trace_step
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt


class WorkflowNodeFactory:
    def __init__(
        self,
        definition: WorkflowDefinitionV2,
        checkpoint_adapter: WorkflowCheckpointAdapter | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.definition = definition
        self.checkpoint_adapter = checkpoint_adapter or WorkflowCheckpointAdapter()
        self.tool_executor = tool_executor or ToolExecutor()

    def create(self, node: WorkflowNodeDefinition):
        async def execute(state: dict[str, Any]) -> dict[str, Any]:
            return await self._execute_node(state, node)

        return execute

    async def _execute_node(
        self,
        state: dict[str, Any],
        node: WorkflowNodeDefinition,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        input_data = self._resolve_mapping(node.input_mapping, state)
        self._before_node(state, node)
        try:
            if node.type == "start":
                output = {}
            elif node.type == "tool":
                output = await self._run_tool_node(node, input_data)
            elif node.type == "agent":
                output = await self._run_agent_node(input_data)
            elif node.type == "llm":
                output = await self._run_llm_node(node, input_data)
            elif node.type == "condition":
                output = self._run_condition_node(state, node)
            elif node.type == "parallel":
                output = await self._run_parallel_node(state, node, input_data)
            elif node.type == "approval":
                output = self._run_approval_node(state, node, input_data)
            elif node.type == "echo":
                output = {"text": str(input_data.get("text", ""))}
                state["answer"] = output["text"]
                state["final_output"] = output
            elif node.type == "final":
                output = self._run_final_node(state)
            else:
                raise WorkflowNodeError(f"unsupported workflow node type: {node.type}")
            self._after_node(state, node, input_data, output, started_at)
            return state
        except GraphInterrupt:
            raise
        except Exception as exc:
            self._fail_node(state, node, input_data, started_at, exc)
            raise

    async def _run_tool_node(
        self,
        node: WorkflowNodeDefinition,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        tool_name = str(node.config.get("tool_name") or input_data.get("tool_name") or "")
        if not tool_name:
            raise WorkflowNodeError(f"tool node missing tool_name: {node.id}")
        result = await self.tool_executor.aexecute(
            ToolCall(name=tool_name, arguments=_clean_none(input_data))
        )
        result_data = result.model_dump()
        if not result.success:
            if node.config.get("fail_open"):
                return {
                    "success": False,
                    "error": result.error,
                    "_tool_result": result_data,
                }
            raise WorkflowNodeError(result.error or f"tool failed: {tool_name}")
        if isinstance(result.result, dict):
            return {**result.result, "_tool_result": result_data}
        return {"result": result.result, "_tool_result": result_data}

    async def _run_agent_node(self, input_data: dict[str, Any]) -> dict[str, Any]:
        from backend.app.agents.factory import AgentRuntimeFactory

        runtime = AgentRuntimeFactory.get_runtime()
        request = AgentRuntimeRequest(
            query=str(input_data.get("query") or ""),
            knowledge_base_id=_optional_int(input_data.get("knowledge_base_id")),
            conversation_id=_optional_int(input_data.get("conversation_id")),
            memory_context=_optional_str(input_data.get("memory_context")),
            metadata=_optional_dict(input_data.get("metadata")),
        )
        arun = getattr(runtime, "arun", None)
        if callable(arun):
            async_run = cast(Callable[[AgentRuntimeRequest], Awaitable[Any]], arun)
            result = await async_run(request)
        else:
            result = await asyncio.to_thread(runtime.run, request)
        return result.model_dump()

    async def _run_llm_node(
        self,
        node: WorkflowNodeDefinition,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = str(input_data.get("prompt") or node.config.get("prompt") or "")
        response = await asyncio.to_thread(
            LLMFactory.get_llm().chat,
            LLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
        )
        return {
            **response.model_dump(),
            "sync_fallback": True,
        }

    def _run_condition_node(
        self,
        state: dict[str, Any],
        node: WorkflowNodeDefinition,
    ) -> dict[str, Any]:
        condition_key = str(node.config.get("condition_key", ""))
        operator = str(node.config.get("operator", "truthy"))
        expected = node.config.get("value")
        value = self._lookup_value(condition_key, state)
        matched = _evaluate_condition(value=value, operator=operator, expected=expected)
        route = "true" if matched else "false"
        routes = node.config.get("routes", {})
        if route not in routes and node.config.get("default_route") is None:
            raise WorkflowConditionError(f"unknown condition route: {route}")
        state["route"] = route if route in routes else str(node.config["default_route"])
        return {"route": state["route"], "matched": matched, "value": value}

    async def _run_parallel_node(
        self,
        state: dict[str, Any],
        node: WorkflowNodeDefinition,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        branches = node.config.get("branches", [])
        if not isinstance(branches, list):
            raise WorkflowNodeError("parallel node branches must be a list")
        semaphore = asyncio.Semaphore(settings.WORKFLOW_MAX_CONCURRENCY)
        fail_fast = bool(node.config.get("fail_fast", settings.WORKFLOW_PARALLEL_FAIL_FAST))

        async def run_branch(branch: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            async with semaphore:
                branch_id = str(branch.get("id") or branch.get("tool_name") or "branch")
                tool_name = str(branch.get("tool_name") or "")
                branch_input = self._resolve_value(branch.get("input", {}), state)
                if not tool_name:
                    return branch_id, {"error": "missing tool_name", "success": False}
                result = await self.tool_executor.aexecute(
                    ToolCall(
                        name=tool_name,
                        arguments=_clean_none(
                            branch_input if isinstance(branch_input, dict) else {}
                        ),
                    )
                )
                if not result.success and fail_fast:
                    raise WorkflowNodeError(result.error or f"parallel branch failed: {branch_id}")
                return branch_id, result.model_dump()

        results = await asyncio.gather(
            *[run_branch(branch) for branch in branches if isinstance(branch, dict)],
            return_exceptions=not fail_fast,
        )
        branch_results: dict[str, Any] = {}
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(str(result))
                continue
            branch_id, branch_output = result
            branch_results[branch_id] = branch_output
        if errors and fail_fast:
            raise WorkflowNodeError(errors[0])
        return {"branch_results": branch_results, "errors": errors}

    def _run_approval_node(
        self,
        state: dict[str, Any],
        node: WorkflowNodeDefinition,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        required_permissions = node.config.get("required_permissions", [])
        check_approval_permission(
            required_permissions=required_permissions
            if isinstance(required_permissions, list)
            else [],
            granted_permissions=state.get("metadata", {}).get("permissions"),
        )
        approval_request = build_approval_request(
            workflow_id=str(state["workflow_id"]),
            run_id=str(state["run_id"]),
            thread_id=str(state["thread_id"]),
            node_id=node.id,
            summary=str(node.config.get("summary") or "Workflow approval required"),
            payload=input_data or state.get("node_outputs", {}),
            required_permissions=required_permissions
            if isinstance(required_permissions, list)
            else [],
            metadata=node.metadata,
        )
        state["approval_request"] = approval_request
        resume_value = interrupt(approval_request)
        approval_result = resume_value if isinstance(resume_value, dict) else {}
        action = str(approval_result.get("action") or "approve")
        if action == "approve":
            route = "approved"
        elif action == "reject":
            route = "rejected"
        elif action == "modify":
            route = "modified"
        else:
            raise WorkflowNodeError(f"unsupported approval action: {action}")
        state["approval_result"] = approval_result
        state["route"] = route
        return {"approval_status": route, "approval_result": approval_result}

    def _run_final_node(self, state: dict[str, Any]) -> dict[str, Any]:
        node_outputs = state.get("node_outputs", {})
        final_output = _find_final_output(node_outputs)
        answer = _extract_answer(final_output)
        state["answer"] = answer
        state["final_output"] = final_output
        state["status"] = "completed"
        return final_output

    def _before_node(self, state: dict[str, Any], node: WorkflowNodeDefinition) -> None:
        step_count = int(state.get("step_count", 0)) + 1
        max_steps = int(state.get("max_steps", self.definition.max_steps))
        if step_count > max_steps:
            raise WorkflowMaxStepsExceeded("workflow max_steps exceeded")
        state["step_count"] = step_count
        state["current_node"] = node.id
        state.setdefault("visited_nodes", []).append(node.id)
        loop_counts = state.setdefault("loop_count_by_node", {})
        loop_counts[node.id] = int(loop_counts.get(node.id, 0)) + 1

    def _after_node(
        self,
        state: dict[str, Any],
        node: WorkflowNodeDefinition,
        input_data: dict[str, Any],
        output: dict[str, Any],
        started_at: float,
    ) -> None:
        state.setdefault("node_outputs", {})[node.id] = output
        state.setdefault("completed_nodes", []).append(node.id)
        state.setdefault("branch_results", {}).update(output.get("branch_results", {}))
        checkpoint_id = self.checkpoint_adapter.save_state(str(state["thread_id"]), state)
        state.setdefault("metadata", {}).setdefault("workflow_runtime", {})[
            "checkpoint_id"
        ] = checkpoint_id
        state.setdefault("metadata", {}).setdefault("workflow_runtime", {})[
            "checkpoint_provider"
        ] = self.checkpoint_adapter.provider
        state.setdefault("trace", []).append(
            make_trace_step(
                step=int(state.get("step_count", 0)),
                workflow_id=str(state["workflow_id"]),
                run_id=str(state["run_id"]),
                thread_id=str(state["thread_id"]),
                node_id=node.id,
                node_type=node.type,
                input_data=input_data,
                output_data=output,
                started_at=started_at,
                checkpoint_id=checkpoint_id,
                approval_status=output.get("approval_status"),
            )
        )

    def _fail_node(
        self,
        state: dict[str, Any],
        node: WorkflowNodeDefinition,
        input_data: dict[str, Any],
        started_at: float,
        exc: Exception,
    ) -> None:
        state["status"] = "failed"
        state.setdefault("errors", []).append(str(exc))
        state.setdefault("failed_nodes", []).append(node.id)
        state.setdefault("trace", []).append(
            make_trace_step(
                step=int(state.get("step_count", 0)),
                workflow_id=str(state["workflow_id"]),
                run_id=str(state["run_id"]),
                thread_id=str(state["thread_id"]),
                node_id=node.id,
                node_type=node.type,
                input_data=input_data,
                output_data={},
                started_at=started_at,
                status="failed",
                error=str(exc),
            )
        )

    def _resolve_mapping(
        self,
        mapping: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if not mapping:
            return {
                "query": state.get("inputs", {}).get("query"),
                "knowledge_base_id": state.get("inputs", {}).get("knowledge_base_id"),
                "metadata": state.get("metadata", {}),
            }
        resolved = self._resolve_value(mapping, state)
        return resolved if isinstance(resolved, dict) else {}

    def _resolve_value(self, value: Any, state: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return self._render_template(value, state)
        if isinstance(value, dict):
            return {key: self._resolve_value(item, state) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item, state) for item in value]
        return value

    def _render_template(self, template: str, state: dict[str, Any]) -> Any:
        exact_match = re.fullmatch(r"\{\{\s*([^}]+)\s*\}\}", template)
        if exact_match:
            return self._lookup_value(exact_match.group(1).strip(), state)

        def replace(match: re.Match[str]) -> str:
            value = self._lookup_value(match.group(1).strip(), state)
            return "" if value is None else str(value)

        return re.sub(r"\{\{\s*([^}]+)\s*\}\}", replace, template)

    def _lookup_value(self, key: str, state: dict[str, Any]) -> Any:
        current: Any
        if key in state.get("inputs", {}):
            current = state["inputs"][key]
        elif key in state:
            current = state[key]
        else:
            if "." not in key:
                return None
            current = state.get("node_outputs", {})

        if "." not in key:
            return current

        current = state.get("node_outputs", {})
        for part in key.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current


def _evaluate_condition(value: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return value is not None
    if operator == "not_equals":
        return value != expected
    if operator == "equals":
        return value == expected
    if operator == "greater_than":
        return value is not None and value > expected
    if operator == "less_than":
        return value is not None and value < expected
    if operator == "in":
        return value in expected if isinstance(expected, list | tuple | set) else False
    if operator == "contains":
        return expected in value if value is not None else False
    if operator == "truthy":
        return bool(value)
    if operator == "falsy":
        return not bool(value)
    if operator == "not_empty":
        return value not in (None, "", [], {})
    raise WorkflowConditionError(f"unsupported condition operator: {operator}")


def _find_final_output(node_outputs: dict[str, Any]) -> dict[str, Any]:
    if isinstance(node_outputs.get("final"), dict):
        return node_outputs["final"]
    for preferred_key in ["agent", "knowledge", "tool"]:
        if isinstance(node_outputs.get(preferred_key), dict):
            return node_outputs[preferred_key]
    if node_outputs:
        last_output = next(reversed(node_outputs.values()))
        return last_output if isinstance(last_output, dict) else {"result": last_output}
    return {}


def _extract_answer(output: dict[str, Any]) -> str | None:
    answer = output.get("answer") or output.get("text") or output.get("result")
    return str(answer) if answer is not None else None


def _clean_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _optional_int(value: Any) -> int | None:
    return int(value) if value not in (None, "") else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _optional_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
