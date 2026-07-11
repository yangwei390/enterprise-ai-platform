from collections import defaultdict, deque

from backend.app.tools import get_tool_registry
from backend.app.workflows.langgraph.errors import WorkflowValidationError
from backend.app.workflows.langgraph.schemas import WorkflowDefinitionV2

SUPPORTED_NODE_TYPES = {
    "start",
    "tool",
    "agent",
    "llm",
    "condition",
    "parallel",
    "approval",
    "echo",
    "final",
}


class WorkflowDefinitionValidator:
    def validate(self, definition: WorkflowDefinitionV2) -> None:
        if not definition.id:
            raise WorkflowValidationError("workflow id is required")
        if definition.max_steps <= 0:
            raise WorkflowValidationError("workflow max_steps must be greater than 0")

        node_ids = [node.id for node in definition.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise WorkflowValidationError("duplicate workflow node id")
        nodes = {node.id: node for node in definition.nodes}
        if definition.entry_node not in nodes:
            raise WorkflowValidationError("entry_node does not exist")

        has_final = any(node.type == "final" for node in definition.nodes)
        if not has_final:
            raise WorkflowValidationError("workflow must contain a final node")

        registry = get_tool_registry()
        edge_pairs: set[tuple[str, str, str | None]] = set()
        for node in definition.nodes:
            if node.type not in SUPPORTED_NODE_TYPES:
                raise WorkflowValidationError(f"unsupported workflow node type: {node.type}")
            if node.type == "tool":
                tool_name = node.config.get("tool_name")
                if tool_name and registry.get_tool(str(tool_name)) is None:
                    raise WorkflowValidationError(f"tool not found: {tool_name}")
            if node.type == "condition":
                if not node.config.get("condition_key"):
                    raise WorkflowValidationError("condition node missing condition_key")
                if not isinstance(node.config.get("routes"), dict):
                    raise WorkflowValidationError("condition node missing routes")
            if node.type == "parallel" and not isinstance(
                node.config.get("branches"), list
            ):
                raise WorkflowValidationError("parallel node missing branches")
            if node.type == "approval":
                routes = node.config.get("routes")
                if not isinstance(routes, dict) or not {"approved", "rejected"} <= set(
                    routes
                ):
                    raise WorkflowValidationError("approval node requires approved/rejected routes")

        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in definition.edges:
            if edge.source not in nodes:
                raise WorkflowValidationError(f"edge source not found: {edge.source}")
            if edge.target not in nodes:
                raise WorkflowValidationError(f"edge target not found: {edge.target}")
            edge_key = (edge.source, edge.target, edge.condition)
            if edge_key in edge_pairs:
                raise WorkflowValidationError("duplicate workflow edge")
            edge_pairs.add(edge_key)
            adjacency[edge.source].append(edge.target)

        reachable = self._reachable(definition.entry_node, adjacency)
        missing = set(node_ids) - reachable
        if missing:
            raise WorkflowValidationError(f"workflow has unreachable nodes: {sorted(missing)}")

    def _reachable(self, entry_node: str, adjacency: dict[str, list[str]]) -> set[str]:
        visited: set[str] = set()
        queue: deque[str] = deque([entry_node])
        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            queue.extend(adjacency.get(node_id, []))
        return visited
