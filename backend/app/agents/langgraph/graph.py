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
    async_mode: bool = False,
) -> Any:
    try:
        graph_module = import_module("langgraph.graph")
    except ImportError as exc:
        raise LangGraphUnavailable("langgraph package is not installed") from exc

    from backend.app.agents.langgraph.state import AgentState

    end_node = graph_module.END
    state_graph = graph_module.StateGraph

    selected_planner_node = planner_node or PlannerNode()
    selected_tool_node = tool_node or ToolNode()
    selected_final_node = final_node or FinalNode()

    graph = state_graph(AgentState)
    if async_mode:
        graph.add_node("planner", selected_planner_node.acall)
        graph.add_node("tool", selected_tool_node.acall)
        graph.add_node("final", selected_final_node.acall)
    else:
        graph.add_node("planner", selected_planner_node)
        graph.add_node("tool", selected_tool_node)
        graph.add_node("final", selected_final_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool")
    graph.add_edge("tool", "final")
    graph.add_edge("final", end_node)
    return graph.compile()
