from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.agents.definition import get_agent_definition_registry
from backend.app.tools.base import ToolDescriptor

TOOL_SCOPE_SOURCE = "agent_definition"
WORKFLOW_TOOL_IDS = {
    "workflow_default_knowledge": "default_agent_workflow_v2",
}


@dataclass(frozen=True)
class ToolScope:
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    allowed_workflows: tuple[str, ...] = field(default_factory=tuple)
    source: str = TOOL_SCOPE_SOURCE
    agent_id: str | None = None
    agent_definition_version: str | None = None
    unrestricted: bool = False

    def allows_tool(self, tool_name: str) -> bool:
        return self.unrestricted or tool_name in self.allowed_tools

    def allows_workflow(self, workflow_id: str) -> bool:
        return self.unrestricted or workflow_id in self.allowed_workflows

    def metadata(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "allowed_workflows": list(self.allowed_workflows),
            "tool_scope_source": self.source,
            "agent_id": self.agent_id,
            "agent_definition_version": self.agent_definition_version,
            "unrestricted": self.unrestricted,
        }


@dataclass(frozen=True)
class ToolPermissionDecision:
    allowed: bool
    tool_name: str
    status: str
    reason: str | None = None
    workflow_id: str | None = None

    def metadata(self, scope: ToolScope) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "workflow_id": self.workflow_id,
            "tool_permission_result": self.status,
            "tool_permission_reason": self.reason,
            **scope.metadata(),
        }


def build_tool_scope(metadata: dict[str, Any] | None) -> ToolScope:
    source_metadata = metadata or {}
    allowlist = source_metadata.get("tool_allowlist")
    workflow_allowlist = source_metadata.get("workflow_allowlist")
    agent_id = _optional_str(source_metadata.get("agent_id"))
    version = _optional_str(source_metadata.get("agent_definition_version"))

    if isinstance(allowlist, list) and isinstance(workflow_allowlist, list):
        return ToolScope(
            allowed_tools=tuple(str(item) for item in allowlist),
            allowed_workflows=tuple(str(item) for item in workflow_allowlist),
            agent_id=agent_id,
            agent_definition_version=version,
        )

    if agent_id is None:
        return ToolScope(source="missing_agent_definition", unrestricted=True)

    try:
        definition = get_agent_definition_registry().get(agent_id)
    except Exception:
        return ToolScope(source="missing_agent_definition", unrestricted=True)

    return ToolScope(
        allowed_tools=tuple(definition.tool_allowlist),
        allowed_workflows=tuple(definition.workflow_allowlist),
        agent_id=definition.id,
        agent_definition_version=definition.version,
    )


def filter_descriptors_by_scope(
    descriptors: list[ToolDescriptor],
    scope: ToolScope,
) -> list[ToolDescriptor]:
    return [descriptor for descriptor in descriptors if scope.allows_tool(descriptor.name)]


def check_tool_permission(
    *,
    tool_name: str,
    arguments: dict[str, Any] | None,
    scope: ToolScope,
) -> ToolPermissionDecision:
    if not scope.allows_tool(tool_name):
        return ToolPermissionDecision(
            allowed=False,
            tool_name=tool_name,
            status="blocked",
            reason="tool_not_allowed",
        )

    workflow_id = workflow_id_for_tool(tool_name=tool_name, arguments=arguments or {})
    if workflow_id and not scope.allows_workflow(workflow_id):
        return ToolPermissionDecision(
            allowed=False,
            tool_name=tool_name,
            status="blocked",
            reason="workflow_not_allowed",
            workflow_id=workflow_id,
        )

    return ToolPermissionDecision(
        allowed=True,
        tool_name=tool_name,
        status="allowed",
        workflow_id=workflow_id,
    )


def workflow_id_for_tool(*, tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name in WORKFLOW_TOOL_IDS:
        return WORKFLOW_TOOL_IDS[tool_name]
    workflow_id = arguments.get("workflow_id")
    return str(workflow_id) if workflow_id else None


def tool_scope_trace(
    *,
    scope: ToolScope,
    visible_tools: list[str],
    selected_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    permission_result: str = "allowed",
    permission_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "allowed_tools": list(scope.allowed_tools),
        "visible_tools": sorted(visible_tools),
        "selected_tools": selected_tools or [],
        "blocked_tools": blocked_tools or [],
        "tool_permission_result": permission_result,
        "tool_permission_reason": permission_reason,
        "tool_scope_source": scope.source,
    }


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
