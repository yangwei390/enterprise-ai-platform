from typing import Any, TypedDict


class WorkflowStateV2(TypedDict, total=False):
    workflow_id: str
    workflow_version: str
    run_id: str
    thread_id: str
    status: str
    inputs: dict[str, Any]
    variables: dict[str, Any]
    node_outputs: dict[str, Any]
    current_node: str | None
    visited_nodes: list[str]
    loop_count_by_node: dict[str, int]
    step_count: int
    max_steps: int
    pending_nodes: list[str]
    completed_nodes: list[str]
    failed_nodes: list[str]
    branch_results: dict[str, Any]
    approval_request: dict[str, Any] | None
    approval_result: dict[str, Any] | None
    final_output: dict[str, Any]
    answer: str | None
    errors: list[str]
    metadata: dict[str, Any]
    trace: list[dict[str, Any]]
    route: str | None
