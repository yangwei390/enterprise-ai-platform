from typing import NotRequired, TypedDict


class AgentState(TypedDict):
    messages: list[dict]
    query: str
    plan: NotRequired[dict]
    tool_calls: list[dict]
    tool_results: list[dict]
    knowledge: NotRequired[dict]
    final_answer: NotRequired[str]
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
        tool_results=[],
        metadata={
            **metadata,
            "runtime": "langgraph",
            "trace": [],
        },
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        memory_context=memory_context,
    )
