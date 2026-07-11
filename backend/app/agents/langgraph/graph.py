from importlib import import_module
from typing import Any

from backend.app.agents.langgraph.nodes import (
    FinalNode,
    ObservationNode,
    PlannerNode,
    ReflectionNode,
    ToolNode,
    route_after_observation,
    route_after_planner,
    route_after_reflection,
)


class LangGraphUnavailable(RuntimeError):
    pass


def build_agent_graph(
    *,
    planner_node: PlannerNode | None = None,
    tool_node: ToolNode | None = None,
    observation_node: ObservationNode | None = None,
    reflection_node: ReflectionNode | None = None,
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
    selected_observation_node = observation_node or ObservationNode()
    selected_reflection_node = reflection_node or ReflectionNode()
    selected_final_node = final_node or FinalNode()

    graph = state_graph(AgentState)
    if async_mode:
        graph.add_node("planner", selected_planner_node.acall)
        graph.add_node("tool", selected_tool_node.acall)
        graph.add_node("observation", selected_observation_node.acall)
        graph.add_node("reflection", selected_reflection_node.acall)
        graph.add_node("final", selected_final_node.acall)
    else:
        graph.add_node("planner", selected_planner_node)
        graph.add_node("tool", selected_tool_node)
        graph.add_node("observation", selected_observation_node)
        graph.add_node("reflection", selected_reflection_node)
        graph.add_node("final", selected_final_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "tool": "tool",
            "reflection": "reflection",
            "final": "final",
        },
    )
    graph.add_edge("tool", "observation")
    graph.add_conditional_edges(
        "observation",
        route_after_observation,
        {
            "planner": "planner",
            "reflection": "reflection",
            "final": "final",
        },
    )
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "planner": "planner",
            "final": "final",
        },
    )
    graph.add_edge("final", end_node)
    return graph.compile()
