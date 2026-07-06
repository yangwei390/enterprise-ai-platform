from backend.app.workflows.base import (
    NodeStatus,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
)
from backend.app.workflows.context import WorkflowContext
from backend.app.workflows.nodes import get_node_executor


class WorkflowRuntime:
    def run(
        self,
        definition: WorkflowDefinition,
        initial_state: dict | None = None,
    ) -> WorkflowResult:
        state = WorkflowState(
            values=initial_state or {},
            current_node_id=definition.start_node_id,
            status=WorkflowStatus.RUNNING,
        )
        context = WorkflowContext(definition=definition, state=state)
        nodes = {node.id: node for node in definition.nodes}

        try:
            while state.current_node_id is not None:
                if context.step_count >= context.max_steps:
                    raise RuntimeError("workflow max_steps exceeded")

                node = nodes.get(state.current_node_id)
                if node is None:
                    raise RuntimeError(f"workflow node not found: {state.current_node_id}")

                context.step_count += 1
                context.add_log(
                    "node_started",
                    {"node_id": node.id, "node_type": node.type},
                )
                state.node_status[node.id] = NodeStatus.RUNNING

                executor = get_node_executor(node.type)
                next_node_id = executor.execute(context, node)

                state.node_status[node.id] = NodeStatus.SUCCESS
                context.add_log(
                    "node_finished",
                    {
                        "node_id": node.id,
                        "node_type": node.type,
                        "next_node_id": next_node_id,
                    },
                )
                state.current_node_id = next_node_id

                if next_node_id is not None and next_node_id not in nodes:
                    raise RuntimeError(f"workflow next node not found: {next_node_id}")

            if state.status != WorkflowStatus.SUCCESS:
                state.status = WorkflowStatus.SUCCESS
        except Exception as exc:
            state.status = WorkflowStatus.FAILED
            state.error = str(exc)
            if state.current_node_id is not None:
                state.node_status[state.current_node_id] = NodeStatus.FAILED
            context.add_log("workflow_failed", {"error": str(exc)})

        return WorkflowResult(
            workflow_id=definition.id,
            status=state.status,
            state=state,
            artifacts=context.artifacts,
            logs=context.logs,
            error=state.error,
        )
