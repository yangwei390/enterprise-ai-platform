from __future__ import annotations

import threading
from collections import OrderedDict
from time import monotonic
from typing import Any

from backend.app.tools.base import ToolResult


class AfterSalesConfirmationCoordinator:
    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._active: dict[str, tuple[str, float]] = {}
        self._completed: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._max_completed = 2048
        self._reservation_ttl_seconds = 60.0

    def reserve(self, session_id: str, operation_id: str) -> bool:
        with self._guard:
            self._cleanup_expired_locked()
            key = (session_id, operation_id)
            if key in self._completed or session_id in self._active:
                return False
            self._active[session_id] = (
                operation_id,
                monotonic() + self._reservation_ttl_seconds,
            )
            return True

    def finalize(self, session_id: str, operation_id: str, *, success: bool) -> None:
        with self._guard:
            active = self._active.get(session_id)
            if active is not None and active[0] == operation_id:
                self._active.pop(session_id, None)
            if success:
                key = (session_id, operation_id)
                self._completed[key] = None
                self._completed.move_to_end(key)
                while len(self._completed) > self._max_completed:
                    self._completed.popitem(last=False)

    def reset(self) -> None:
        with self._guard:
            self._active.clear()
            self._completed.clear()

    def _cleanup_expired_locked(self) -> None:
        now = monotonic()
        expired = [
            session_id
            for session_id, (_, expires_at) in self._active.items()
            if expires_at <= now
        ]
        for session_id in expired:
            self._active.pop(session_id, None)


confirmation_coordinator = AfterSalesConfirmationCoordinator()


def reserve_after_sales_confirmation(
    *,
    state: dict[str, Any],
    pending: dict[str, Any],
) -> str | None:
    latest_user = _latest_user_message(state)
    if latest_user is None:
        return "after_sales_user_confirmation_missing"
    normalized = latest_user.strip()
    if any(term in normalized.casefold() for term in ("忽略系统", "confirmed=true")):
        return "after_sales_unsafe_injection"
    if normalized in {"取消", "不确认", "不要提交"}:
        return "after_sales_confirmation_cancelled"
    if normalized not in {"确认", "确认提交", "同意提交", "可以提交"}:
        return "after_sales_explicit_confirmation_required"
    conversation_id = state.get("conversation_id")
    if conversation_id is None:
        return "after_sales_conversation_required"
    operation_id = pending.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        return "after_sales_confirmation_missing"
    session_id = f"conversation:{conversation_id}"
    if not confirmation_coordinator.reserve(session_id, operation_id):
        return "after_sales_confirmation_in_progress"
    return None


def finalize_after_sales_confirmation(
    *,
    state: dict[str, Any],
    operation_id: str,
    result: ToolResult,
) -> None:
    conversation_id = state.get("conversation_id")
    if conversation_id is None:
        return
    confirmation_coordinator.finalize(
        f"conversation:{conversation_id}",
        operation_id,
        success=result.success,
    )


def _latest_user_message(state: dict[str, Any]) -> str | None:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") not in {"user", "human"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None
