import re
from time import perf_counter
from typing import Any, Protocol

from backend.app.tools import ToolCall, ToolExecutor, ToolResult
from pydantic import BaseModel, Field, field_validator


class WorkflowV1Node(BaseModel):
    id: str
    type: str
    tool_name: str | None = None
    input: dict = Field(default_factory=dict)


class WorkflowV1Definition(BaseModel):
    id: str = "default_knowledge_workflow"
    name: str = "default_knowledge_workflow"
    nodes: list[WorkflowV1Node]


class WorkflowRunRequest(BaseModel):
    workflow_id: str | None = "default_knowledge_workflow"
    query: str
    knowledge_base_id: int | None = None
    inputs: dict = Field(default_factory=dict)
    thread_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    definition: WorkflowV1Definition | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query不能为空")
        return query


class WorkflowTraceStep(BaseModel):
    step: int
    node_id: str
    node_type: str
    input: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    duration_ms: float = 0
    status: str = "success"
    error: str | None = None


class WorkflowRunResult(BaseModel):
    answer: str | None = None
    output: dict = Field(default_factory=dict)
    node_outputs: dict = Field(default_factory=dict)
    trace: list[WorkflowTraceStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ToolExecutorProtocol(Protocol):
    def execute(self, tool_call: ToolCall) -> ToolResult:
        ...


class AgentRuntimeProtocol(Protocol):
    def run(self, request: Any) -> Any:
        ...


class WorkflowRuntimeV1:
    def __init__(
        self,
        tool_executor: ToolExecutorProtocol | None = None,
        agent_runtime: AgentRuntimeProtocol | None = None,
    ) -> None:
        if agent_runtime is None:
            from backend.app.agents.runtime import AgentRuntime

            agent_runtime = AgentRuntime()
        self.tool_executor = tool_executor or ToolExecutor()
        self.agent_runtime = agent_runtime

    def run(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        definition = request.definition or self._default_definition(request.workflow_id)
        state = {
            "query": request.query,
            "knowledge_base_id": request.knowledge_base_id,
            **request.inputs,
        }
        node_outputs: dict[str, Any] = {}
        trace: list[WorkflowTraceStep] = []
        metadata = {
            "workflow_id": definition.id,
            "workflow_name": definition.name,
            "failed": False,
        }

        for index, node in enumerate(definition.nodes, start=1):
            resolved_input = self._resolve_value(node.input, state, node_outputs)
            started_at = perf_counter()
            try:
                output = self._run_node(node=node, resolved_input=resolved_input)
                node_outputs[node.id] = output
                state[node.id] = output
                trace.append(
                    self._trace_step(
                        index=index,
                        node=node,
                        input_data=resolved_input,
                        output_data=output,
                        started_at=started_at,
                    )
                )
            except Exception as exc:
                metadata["failed"] = True
                metadata["error"] = str(exc)
                trace.append(
                    self._trace_step(
                        index=index,
                        node=node,
                        input_data=resolved_input,
                        output_data={},
                        started_at=started_at,
                        status="failed",
                        error=str(exc),
                    )
                )
                break

        final_output = self._find_final_output(node_outputs)
        return WorkflowRunResult(
            answer=self._extract_answer(final_output),
            output=final_output,
            node_outputs=node_outputs,
            trace=trace,
            metadata=metadata,
        )

    def _run_node(self, node: WorkflowV1Node, resolved_input: dict) -> dict:
        if node.type == "tool":
            return self._run_tool_node(node, resolved_input)
        if node.type == "agent":
            return self._run_agent_node(resolved_input)
        if node.type == "echo":
            return {"text": str(resolved_input.get("text", ""))}
        raise ValueError(f"Unsupported workflow node type: {node.type}")

    def _run_tool_node(self, node: WorkflowV1Node, resolved_input: dict) -> dict:
        tool_name = node.tool_name or str(resolved_input.get("tool_name", ""))
        if not tool_name:
            raise ValueError(f"tool node {node.id} missing tool_name")

        result = self.tool_executor.execute(
            ToolCall(name=tool_name, arguments=self._clean_null_values(resolved_input))
        )
        if not result.success:
            raise RuntimeError(result.error or f"tool failed: {tool_name}")

        if isinstance(result.result, dict):
            return {
                **result.result,
                "_tool_result": result.model_dump(),
            }
        return {
            "result": result.result,
            "_tool_result": result.model_dump(),
        }

    def _run_agent_node(self, resolved_input: dict) -> dict:
        from backend.app.agents.state import AgentRuntimeRequest

        query = str(resolved_input.get("query", ""))
        result = self.agent_runtime.run(
            AgentRuntimeRequest(
                query=query,
                knowledge_base_id=self._optional_int(resolved_input.get("knowledge_base_id")),
                conversation_id=self._optional_int(resolved_input.get("conversation_id")),
                memory_context=self._optional_str(resolved_input.get("memory_context")),
                metadata=self._optional_dict(resolved_input.get("metadata")),
            )
        )
        return result.model_dump()

    def _default_definition(self, workflow_id: str | None) -> WorkflowV1Definition:
        if workflow_id not in (None, "default_knowledge_workflow"):
            raise ValueError(f"Unsupported workflow_id: {workflow_id}")
        return WorkflowV1Definition(
            id="default_knowledge_workflow",
            name="default_knowledge_workflow",
            nodes=[
                WorkflowV1Node(
                    id="knowledge",
                    type="tool",
                    tool_name="knowledge_search",
                    input={
                        "query": "{{query}}",
                        "knowledge_base_id": "{{knowledge_base_id}}",
                    },
                ),
                WorkflowV1Node(
                    id="final",
                    type="echo",
                    input={
                        "text": "{{knowledge.answer}}",
                    },
                ),
            ],
        )

    def _resolve_value(
        self,
        value: Any,
        state: dict[str, Any],
        node_outputs: dict[str, Any],
    ) -> Any:
        if isinstance(value, str):
            return self._render_template(value, state, node_outputs)
        if isinstance(value, dict):
            return {
                key: self._resolve_value(item, state, node_outputs)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_value(item, state, node_outputs) for item in value]
        return value

    def _render_template(
        self,
        template: str,
        state: dict[str, Any],
        node_outputs: dict[str, Any],
    ) -> Any:
        exact_match = re.fullmatch(r"\{\{\s*([^}]+)\s*\}\}", template)
        if exact_match:
            return self._lookup_value(exact_match.group(1).strip(), state, node_outputs)

        def replace(match: re.Match[str]) -> str:
            value = self._lookup_value(match.group(1).strip(), state, node_outputs)
            return "" if value is None else str(value)

        return re.sub(r"\{\{\s*([^}]+)\s*\}\}", replace, template)

    def _lookup_value(
        self,
        key: str,
        state: dict[str, Any],
        node_outputs: dict[str, Any],
    ) -> Any:
        if "." not in key:
            return state.get(key)

        current: Any = node_outputs
        for part in key.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _trace_step(
        self,
        index: int,
        node: WorkflowV1Node,
        input_data: dict,
        output_data: dict,
        started_at: float,
        status: str = "success",
        error: str | None = None,
    ) -> WorkflowTraceStep:
        return WorkflowTraceStep(
            step=index,
            node_id=node.id,
            node_type=node.type,
            input=input_data,
            output=output_data,
            duration_ms=round((perf_counter() - started_at) * 1000, 2),
            status=status,
            error=error,
        )

    def _find_final_output(self, node_outputs: dict[str, Any]) -> dict:
        if not node_outputs:
            return {}
        if isinstance(node_outputs.get("final"), dict):
            return node_outputs["final"]
        last_output = next(reversed(node_outputs.values()))
        return last_output if isinstance(last_output, dict) else {"result": last_output}

    def _extract_answer(self, output: dict) -> str | None:
        answer = output.get("answer") or output.get("text") or output.get("result")
        return str(answer) if answer is not None else None

    def _clean_null_values(self, value: dict) -> dict:
        return {key: item for key, item in value.items() if item is not None}

    def _optional_int(self, value: Any) -> int | None:
        return int(value) if value not in (None, "") else None

    def _optional_str(self, value: Any) -> str | None:
        return str(value) if value not in (None, "") else None

    def _optional_dict(self, value: Any) -> dict:
        return value if isinstance(value, dict) else {}
