from typing import NotRequired, TypedDict

from backend.app.agents.langgraph.budget import AgentExecutionBudget


class AgentState(TypedDict):
    messages: list[dict]
    query: str
    plan: NotRequired[dict]
    tool_calls: list[dict]
    pending_tool_calls: list[dict]
    tool_results: list[dict]
    observations: list[dict]
    knowledge: NotRequired[dict]
    final_answer: NotRequired[str]
    current_action: NotRequired[str]
    step_count: int
    llm_call_count: int
    tool_call_count: int
    reflection_count: int
    same_tool_repeat_count: int
    last_tool_name: NotRequired[str | None]
    last_tool_arguments_hash: NotRequired[str | None]
    loop_status: str
    termination_reason: NotRequired[str | None]
    budget: dict
    metadata: dict
    conversation_id: NotRequired[int | None]
    knowledge_base_id: NotRequired[int | None]
    memory_context: NotRequired[str | None]


def create_initial_state(
    *,
    query: str,
    conversation_id: int | None,
    knowledge_base_id: int | None,
    memory_context: str | None,
    metadata: dict,
) -> AgentState:
    return AgentState(
        messages=[{"role": "user", "content": query}],
        query=query,
        tool_calls=[],
        pending_tool_calls=[],
        tool_results=[],
        observations=[],
        step_count=0,
        llm_call_count=0,
        tool_call_count=0,
        reflection_count=0,
        same_tool_repeat_count=0,
        last_tool_name=None,
        last_tool_arguments_hash=None,
        loop_status="running",
        termination_reason=None,
        budget=AgentExecutionBudget.from_settings().model_dump(),
        metadata={
            **metadata,
            "runtime": "langgraph",
            "trace": [],
            "agent_loop": {
                "enabled": True,
                "planner_strategy": "native_tool_calling",
                "planner_fallback_used": False,
                "steps": 0,
                "llm_calls": 0,
                "tool_calls": 0,
                "reflections": 0,
                "loop_iterations": 0,
                "termination_reason": None,
                "budget_exceeded": False,
                "same_tool_repeat_limit_triggered": False,
                "duration_ms": 0,
            },
            "tool_calling": {
                "native_supported": False,
                "native_used": False,
                "available_tool_count": 0,
                "registry_version": 0,
            },
            "reflection": {
                "enabled": True,
                "triggered": False,
                "count": 0,
                "last_reason": None,
            },
        },
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        memory_context=memory_context,
    )
