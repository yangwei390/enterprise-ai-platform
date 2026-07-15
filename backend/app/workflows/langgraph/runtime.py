import asyncio
import uuid
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any, cast

from backend.app.logger import logger
from backend.app.workflows.langgraph.checkpoint import (
    WorkflowCheckpointAdapter,
    build_langgraph_checkpointer,
)
from backend.app.workflows.langgraph.definition import get_workflow_definition_v2
from backend.app.workflows.langgraph.errors import (
    WorkflowAlreadyCompletedError,
    WorkflowNotInterruptedError,
    WorkflowResumeError,
)
from backend.app.workflows.langgraph.graph import WorkflowGraphBuilder
from backend.app.workflows.langgraph.schemas import (
    WorkflowDefinitionV2,
    WorkflowResumeRequest,
    WorkflowRunRequestV2,
    WorkflowRunResultV2,
)
from backend.app.workflows.langgraph.validator import WorkflowDefinitionValidator
from langgraph.types import Command


class LangGraphWorkflowRuntime:
    def __init__(
        self,
        graph_builder: WorkflowGraphBuilder | None = None,
        checkpoint_adapter: WorkflowCheckpointAdapter | None = None,
    ) -> None:
        self.graph_builder = graph_builder or WorkflowGraphBuilder()
        self.checkpoint_adapter = checkpoint_adapter or WorkflowCheckpointAdapter()
        self.validator = WorkflowDefinitionValidator()
        self._graphs: dict[str, Any] = {}
        self._definitions: dict[str, WorkflowDefinitionV2] = {}
        self._states: dict[str, dict[str, Any]] = {}

    def run(self, request: WorkflowRunRequestV2) -> WorkflowRunResultV2:
        return asyncio.run(self.arun(request))

    async def arun(self, request: WorkflowRunRequestV2) -> WorkflowRunResultV2:
        started_at = perf_counter()
        definition: WorkflowDefinitionV2 | None = None
        state: dict[str, Any] | None = None
        try:
            definition = get_workflow_definition_v2(request.workflow_id, request.definition)
            self.validator.validate(definition)
            graph = self._get_graph(definition)
            run_id = str(uuid.uuid4())
            thread_id = request.thread_id or str(uuid.uuid4())
            state = self._initial_state(
                definition=definition,
                request=request,
                run_id=run_id,
                thread_id=thread_id,
            )
            async with asyncio.timeout(definition.timeout_seconds):
                result_state = await graph.ainvoke(
                    cast(Any, state),
                    config={"configurable": {"thread_id": thread_id}},
                )
            return self._to_result(
                definition=definition,
                state=result_state,
                started_at=started_at,
                resumed=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Workflow V2 failed")
            if state is not None and definition is not None:
                state["status"] = "failed"
                state.setdefault("errors", []).append(str(exc))
                return self._to_result(
                    definition=definition,
                    state=state,
                    started_at=started_at,
                    resumed=False,
                )
            raise

    async def aresume(self, request: WorkflowResumeRequest) -> WorkflowRunResultV2:
        started_at = perf_counter()
        state = self._states.get(request.thread_id) or self.checkpoint_adapter.load_state(
            request.thread_id
        )
        if state is None:
            raise WorkflowResumeError(f"workflow thread not found: {request.thread_id}")
        if request.run_id and state.get("run_id") != request.run_id:
            raise WorkflowResumeError("workflow run_id does not match thread state")
        if state.get("status") == "completed":
            raise WorkflowAlreadyCompletedError("workflow already completed")
        if state.get("status") != "interrupted":
            raise WorkflowNotInterruptedError("workflow is not interrupted")

        workflow_id = str(state.get("workflow_id") or request.workflow_id)
        definition = self._definitions.get(workflow_id) or get_workflow_definition_v2(workflow_id)
        graph = self._get_graph(definition)
        resume_payload = request.command.model_dump()
        result_state = await graph.ainvoke(
            Command(resume=resume_payload),
            config={"configurable": {"thread_id": request.thread_id}},
        )
        return self._to_result(
            definition=definition,
            state=result_state,
            started_at=started_at,
            resumed=True,
        )

    def resume(self, request: WorkflowResumeRequest) -> WorkflowRunResultV2:
        return asyncio.run(self.aresume(request))

    async def astream(
        self,
        request: WorkflowRunRequestV2,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "workflow_started", "data": request.model_dump()}
        result = await self.arun(request)
        for trace in result.trace:
            yield {"event": "node_completed", "data": trace}
        event = "approval_required" if result.status == "interrupted" else "workflow_completed"
        yield {"event": event, "data": result.model_dump()}

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        return self._states.get(thread_id) or self.checkpoint_adapter.load_state(thread_id)

    def list_definitions(self) -> list[WorkflowDefinitionV2]:
        from backend.app.workflows.langgraph.definition import list_workflow_definitions_v2

        return list_workflow_definitions_v2()

    def _get_graph(self, definition: WorkflowDefinitionV2):
        graph = self._graphs.get(definition.id)
        if graph is None:
            graph = self.graph_builder.compile(
                definition,
                checkpointer=build_langgraph_checkpointer(),
            )
            self._graphs[definition.id] = graph
            self._definitions[definition.id] = definition
        return graph

    def _initial_state(
        self,
        *,
        definition: WorkflowDefinitionV2,
        request: WorkflowRunRequestV2,
        run_id: str,
        thread_id: str,
    ) -> dict[str, Any]:
        inputs = {
            "query": request.query,
            "knowledge_base_id": request.knowledge_base_id,
            **request.inputs,
        }
        metadata = {
            **request.metadata,
            "workflow_runtime": {
                "runtime": "langgraph_v2",
                "workflow_version": definition.version,
                "thread_id": thread_id,
                "run_id": run_id,
                "step_count": 0,
                "max_steps": definition.max_steps,
                "checkpoint_enabled": definition.checkpoint_enabled,
                "checkpoint_provider": self.checkpoint_adapter.provider,
                "resumed": False,
                "interrupted": False,
                "approval_required": False,
                "duration_ms": 0,
                "failed": False,
                "error": None,
            },
        }
        return {
            "workflow_id": definition.id,
            "workflow_version": definition.version,
            "run_id": run_id,
            "thread_id": thread_id,
            "status": "running",
            "inputs": inputs,
            "variables": {},
            "node_outputs": {},
            "current_node": None,
            "visited_nodes": [],
            "loop_count_by_node": {},
            "step_count": 0,
            "max_steps": definition.max_steps,
            "pending_nodes": [],
            "completed_nodes": [],
            "failed_nodes": [],
            "branch_results": {},
            "approval_request": None,
            "approval_result": None,
            "final_output": {},
            "answer": None,
            "errors": [],
            "metadata": metadata,
            "trace": [],
            "route": None,
        }

    def _to_result(
        self,
        *,
        definition: WorkflowDefinitionV2,
        state: dict[str, Any],
        started_at: float,
        resumed: bool,
    ) -> WorkflowRunResultV2:
        interrupts = state.get("__interrupt__")
        interrupt_payload = None
        status = str(state.get("status") or "completed")
        if interrupts:
            first_interrupt = interrupts[0]
            interrupt_payload = getattr(first_interrupt, "value", first_interrupt)
            status = "interrupted"
            state["status"] = "interrupted"
            state["approval_request"] = interrupt_payload

        metadata = dict(state.get("metadata", {}))
        workflow_metadata = metadata.setdefault("workflow_runtime", {})
        workflow_metadata.update(
            {
                "runtime": "langgraph_v2",
                "workflow_version": definition.version,
                "thread_id": state.get("thread_id"),
                "run_id": state.get("run_id"),
                "step_count": state.get("step_count", 0),
                "max_steps": definition.max_steps,
                "checkpoint_enabled": definition.checkpoint_enabled,
                "checkpoint_provider": self.checkpoint_adapter.provider,
                "resumed": resumed,
                "interrupted": status == "interrupted",
                "approval_required": status == "interrupted",
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                "failed": status == "failed",
                "error": "; ".join(state.get("errors", [])) or None,
            }
        )
        state["metadata"] = metadata
        self._states[str(state["thread_id"])] = state
        self.checkpoint_adapter.save_state(str(state["thread_id"]), state)
        workflow_metadata["checkpoint_provider"] = self.checkpoint_adapter.provider
        return WorkflowRunResultV2(
            workflow_id=str(state["workflow_id"]),
            run_id=str(state["run_id"]),
            thread_id=str(state["thread_id"]),
            status=status,
            answer=state.get("answer"),
            output=state.get("final_output", {}),
            node_outputs=state.get("node_outputs", {}),
            trace=state.get("trace", []),
            metadata=metadata,
            interrupt=interrupt_payload if isinstance(interrupt_payload, dict) else None,
        )
