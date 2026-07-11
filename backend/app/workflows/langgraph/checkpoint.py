from typing import Any

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.memory.factory import MemoryFactory
from langgraph.checkpoint.memory import InMemorySaver


class WorkflowCheckpointAdapter:
    _fallback_store: dict[str, dict[str, Any]] = {}

    def __init__(self) -> None:
        self.manager = MemoryFactory.get_checkpoint_manager()
        self.provider = settings.WORKFLOW_CHECKPOINT_PROVIDER

    def save_state(self, thread_id: str, state: dict[str, Any]) -> str:
        checkpoint_id = self.checkpoint_id(thread_id)
        serializable_state = self._make_json_safe(state)
        try:
            self.manager.save(checkpoint_id, serializable_state)
            self.provider = settings.WORKFLOW_CHECKPOINT_PROVIDER
        except Exception as exc:
            if not settings.WORKFLOW_CHECKPOINT_FAIL_OPEN:
                raise
            logger.warning(
                f"Workflow checkpoint save failed, using memory fallback | "
                f"checkpoint={checkpoint_id} | error={exc}"
            )
            self._fallback_store[checkpoint_id] = serializable_state
            self.provider = "memory"
        return checkpoint_id

    def load_state(self, thread_id: str) -> dict[str, Any] | None:
        checkpoint_id = self.checkpoint_id(thread_id)
        try:
            return self.manager.load(checkpoint_id) or self._fallback_store.get(checkpoint_id)
        except Exception as exc:
            if not settings.WORKFLOW_CHECKPOINT_FAIL_OPEN:
                raise
            logger.warning(
                f"Workflow checkpoint load failed, using memory fallback | "
                f"checkpoint={checkpoint_id} | error={exc}"
            )
            self.provider = "memory"
            return self._fallback_store.get(checkpoint_id)

    def delete_state(self, thread_id: str) -> None:
        checkpoint_id = self.checkpoint_id(thread_id)
        self._fallback_store.pop(checkpoint_id, None)
        try:
            self.manager.delete(checkpoint_id)
        except Exception:
            if not settings.WORKFLOW_CHECKPOINT_FAIL_OPEN:
                raise

    def list(self) -> list[str]:
        try:
            return [*self.manager.list(), *self._fallback_store.keys()]
        except Exception:
            if not settings.WORKFLOW_CHECKPOINT_FAIL_OPEN:
                raise
            return list(self._fallback_store)

    def checkpoint_id(self, thread_id: str) -> str:
        return f"workflow:{thread_id}"

    def _make_json_safe(self, value: Any, seen: set[int] | None = None) -> Any:
        seen = seen or set()
        value_id = id(value)
        if isinstance(value, dict | list | tuple | set):
            if value_id in seen:
                return "<circular-reference>"
            seen.add(value_id)
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item, seen) for key, item in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(item, seen) for item in value]
        if isinstance(value, tuple):
            return [self._make_json_safe(item, seen) for item in value]
        if isinstance(value, set):
            return [self._make_json_safe(item, seen) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        if hasattr(value, "value"):
            return self._make_json_safe(value.value, seen)
        if hasattr(value, "model_dump"):
            return self._make_json_safe(value.model_dump(), seen)
        return str(value)


def build_langgraph_checkpointer():
    if not settings.WORKFLOW_CHECKPOINT_ENABLED:
        return None
    return InMemorySaver()
