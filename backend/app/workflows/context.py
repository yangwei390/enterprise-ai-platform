from typing import Any

from backend.app.workflows.base import (
    WorkflowArtifact,
    WorkflowDefinition,
    WorkflowState,
)


class WorkflowContext:
    def __init__(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState | None = None,
        max_steps: int = 20,
    ) -> None:
        self.definition = definition
        self.state = state or WorkflowState()
        self.artifacts: list[WorkflowArtifact] = []
        self.logs: list[dict] = []
        self.max_steps = max_steps
        self.step_count = 0

    def add_log(self, event: str, data: dict | None = None) -> None:
        self.logs.append({"event": event, "data": data or {}})

    def set_value(self, key: str, value: Any) -> None:
        self.state.values[key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.state.values.get(key, default)

    def add_artifact(
        self,
        key: str,
        value: dict | str | int | float | list | None,
        metadata: dict | None = None,
    ) -> None:
        self.artifacts.append(
            WorkflowArtifact(key=key, value=value, metadata=metadata or {})
        )
