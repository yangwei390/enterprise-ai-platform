from importlib import import_module
from typing import Any

from backend.app.agents.langgraph.nodes import FinalNode, PlannerNode, ToolNode


class LangGraphUnavailable(RuntimeError):
    pass


def build_agent_graph(
    *,
    planner_node: PlannerNode | None = None,
    tool_node: ToolNode | None = None,
    final_node: FinalNode | None = None,
) -> Any:
    try:
        graph_module = import_module("langgraph.graph")
    except ImportError as exc:
        raise LangGraphUnavailable("langgraph package is not installed") from exc

    from backend.app.agents.langgraph.state import AgentState

    end_node = graph_module.END
    state_graph = graph_module.StateGraph

    graph = state_graph(AgentState)
    graph.add_node("planner", planner_node or PlannerNode())
    graph.add_node("tool", tool_node or ToolNode())
    graph.add_node("final", final_node or FinalNode())
    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool")
    graph.add_edge("tool", "final")
    graph.add_edge("final", end_node)
    return graph.compile()
