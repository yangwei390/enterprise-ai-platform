from time import perf_counter
from typing import Any


def make_trace_step(
    *,
    step: int,
    workflow_id: str,
    run_id: str,
    thread_id: str,
    node_id: str,
    node_type: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    started_at: float,
    status: str = "success",
    error: str | None = None,
    async_execution: bool = True,
    interrupted: bool = False,
    approval_status: str | None = None,
    branch_id: str | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "workflow_id": workflow_id,
        "run_id": run_id,
        "thread_id": thread_id,
        "node_id": node_id,
        "node_type": node_type,
        "input": input_data,
        "output": output_data,
        "status": status,
        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "async_execution": async_execution,
        "retry_count": 0,
        "timeout": False,
        "cache_hit": False,
        "checkpoint_id": checkpoint_id,
        "interrupted": interrupted,
        "approval_status": approval_status,
        "branch_id": branch_id,
        "error": error,
    }
