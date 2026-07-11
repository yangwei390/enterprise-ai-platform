from collections.abc import Hashable
from typing import Any, cast

from backend.app.workflows.langgraph.checkpoint import build_langgraph_checkpointer
from backend.app.workflows.langgraph.nodes import WorkflowNodeFactory
from backend.app.workflows.langgraph.schemas import WorkflowDefinitionV2
from backend.app.workflows.langgraph.state import WorkflowStateV2
from langgraph.graph import END, StateGraph


class WorkflowGraphBuilder:
    def __init__(self, node_factory: WorkflowNodeFactory | None = None) -> None:
        self.node_factory = node_factory

    def build(self, definition: WorkflowDefinitionV2, checkpointer: Any = None):
        graph = StateGraph(WorkflowStateV2)
        node_factory = self.node_factory or WorkflowNodeFactory(definition)
        node_ids = {node.id for node in definition.nodes}
        conditional_sources = {
            edge.source for edge in definition.edges if edge.condition is not None
        }

        for node in definition.nodes:
            graph.add_node(node.id, cast(Any, node_factory.create(node)))
        graph.set_entry_point(definition.entry_node)

        for source in conditional_sources:
            routes = {
                str(edge.condition): edge.target
                for edge in definition.edges
                if edge.source == source and edge.condition is not None
            }
            graph.add_conditional_edges(source, _route, cast(dict[Hashable, str], routes))

        for edge in definition.edges:
            if edge.condition is None and edge.source not in conditional_sources:
                graph.add_edge(edge.source, edge.target)

        sources = {edge.source for edge in definition.edges}
        for node_id in node_ids:
            if node_id not in sources:
                graph.add_edge(node_id, END)

        return graph.compile(checkpointer=checkpointer or build_langgraph_checkpointer())

    def compile(self, definition: WorkflowDefinitionV2, checkpointer: Any = None):
        return self.build(definition, checkpointer=checkpointer)


def _route(state: dict[str, Any]) -> str:
    return str(state.get("route") or "default")
