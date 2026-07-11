from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.config.settings import settings
from backend.app.workflows.langgraph.errors import WorkflowPermissionError


def build_approval_request(
    *,
    workflow_id: str,
    run_id: str,
    thread_id: str,
    node_id: str,
    summary: str,
    payload: dict[str, Any],
    required_permissions: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    expires_at = datetime.now(UTC) + timedelta(
        seconds=settings.WORKFLOW_APPROVAL_DEFAULT_TIMEOUT_SECONDS
    )
    return {
        "workflow_id": workflow_id,
        "run_id": run_id,
        "thread_id": thread_id,
        "node_id": node_id,
        "action": "approval_required",
        "summary": summary,
        "payload": payload,
        "required_permissions": required_permissions,
        "expires_at": expires_at.isoformat(),
        "metadata": metadata,
    }


def check_approval_permission(
    *,
    required_permissions: list[str],
    granted_permissions: list[str] | None,
) -> None:
    if not settings.WORKFLOW_APPROVAL_PERMISSION_ENFORCEMENT:
        return
    if not required_permissions:
        return
    granted = set(granted_permissions or [])
    missing = [item for item in required_permissions if item not in granted]
    if missing:
        raise WorkflowPermissionError(
            f"missing workflow approval permissions: {', '.join(missing)}"
        )
