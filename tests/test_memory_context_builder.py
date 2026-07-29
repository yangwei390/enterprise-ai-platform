from backend.app.memory.context_builder import build_bounded_message_context


def test_context_builder_keeps_recent_turns_and_summarizes_older_turns() -> None:
    messages = [
        message
        for turn in range(1, 6)
        for message in (
            {
                "role": "user",
                "content": f"question-{turn}",
                "turn_id": f"turn-{turn}",
            },
            {
                "role": "assistant",
                "content": f"answer-{turn}",
                "turn_id": f"turn-{turn}",
            },
        )
    ]

    context = build_bounded_message_context(messages, max_turns=3)

    assert context[0]["role"] == "system"
    assert "question-1" in context[0]["content"]
    assert "question-2" in context[0]["content"]
    assert [item["content"] for item in context[1:]] == [
        "question-3",
        "answer-3",
        "question-4",
        "answer-4",
        "question-5",
        "answer-5",
    ]
    assert len(messages) == 10


def test_context_builder_compresses_only_historical_tool_messages() -> None:
    messages = [
        {
            "role": "user",
            "content": f"question-{turn}",
            "turn_id": f"turn-{turn}",
        }
        for turn in range(1, 5)
    ]
    messages.extend(
        [
            {
                "role": "tool",
                "name": "search",
                "content": "x" * 1000,
                "turn_id": "turn-3",
            },
            {
                "role": "tool",
                "name": "search",
                "content": "y" * 1000,
                "turn_id": "turn-4",
            },
        ]
    )

    context = build_bounded_message_context(
        messages,
        max_turns=3,
        historical_tool_max_chars=50,
    )
    tool_messages = [item for item in context if item["role"] == "tool"]

    assert len(tool_messages[0]["content"]) == 50
    assert len(tool_messages[1]["content"]) == 1000
