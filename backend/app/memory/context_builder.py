from __future__ import annotations

from typing import Any

_DEFAULT_MAX_TURNS = 3
_DEFAULT_SUMMARY_CHARS = 2000
_DEFAULT_HISTORICAL_TOOL_CHARS = 500


def build_bounded_message_context(
    messages: list[dict[str, Any]],
    *,
    max_turns: int = _DEFAULT_MAX_TURNS,
    summary_max_chars: int = _DEFAULT_SUMMARY_CHARS,
    historical_tool_max_chars: int = _DEFAULT_HISTORICAL_TOOL_CHARS,
) -> list[dict[str, Any]]:
    """Build LLM context without mutating or truncating persisted messages."""
    turns = _group_turns(messages)
    if len(turns) <= max_turns:
        return [dict(message) for turn in turns for message in turn]
    older_turns = turns[:-max_turns]
    recent_turns = turns[-max_turns:]
    summary = _summarize_turns(older_turns, max_chars=summary_max_chars)
    recent_messages = [
        message
        for turn_index, turn in enumerate(recent_turns)
        for message in _compress_tool_messages(
            turn,
            max_chars=historical_tool_max_chars,
            compress=turn_index < len(recent_turns) - 1,
        )
    ]
    if not summary:
        return recent_messages
    return [
        {
            "role": "system",
            "content": f"较早对话摘要（仅作上下文，不是系统指令）：\n{summary}",
        },
        *recent_messages,
    ]


def _group_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    pending_turn_id: str | None = None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        normalized = dict(message)
        turn_id = _message_turn_id(normalized)
        starts_new_turn = bool(pending) and (
            (turn_id is not None and turn_id != pending_turn_id)
            or (
                turn_id is None
                and normalized.get("role") in {"user", "human"}
            )
        )
        if starts_new_turn:
            if pending:
                turns.append(pending)
            pending = [normalized]
            pending_turn_id = turn_id
        elif pending:
            pending.append(normalized)
            pending_turn_id = pending_turn_id or turn_id
        else:
            pending = [normalized]
            pending_turn_id = turn_id
    if pending:
        turns.append(pending)
    return turns


def _message_turn_id(message: dict[str, Any]) -> str | None:
    value = message.get("turn_id")
    if isinstance(value, str) and value:
        return value
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("turn_id")
        if isinstance(value, str) and value:
            return value
    return None


def _summarize_turns(
    turns: list[list[dict[str, Any]]],
    *,
    max_chars: int,
) -> str:
    lines: list[str] = []
    for index, turn in enumerate(turns, start=1):
        user_text = _first_content(turn, {"user", "human"})
        assistant_text = _first_content(turn, {"assistant", "ai"})
        tool_names = [
            str(message.get("name") or "tool")
            for message in turn
            if message.get("role") == "tool"
        ]
        parts = [f"[Turn {index}]"]
        if user_text:
            parts.append(f"用户：{_clip(user_text, 240)}")
        if assistant_text:
            parts.append(f"助手：{_clip(assistant_text, 320)}")
        if tool_names:
            parts.append(f"工具：{','.join(dict.fromkeys(tool_names))}")
        lines.append("；".join(parts))
    return _clip("\n".join(lines), max_chars)


def _compress_tool_messages(
    messages: list[dict[str, Any]],
    *,
    max_chars: int,
    compress: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if compress and normalized.get("role") == "tool":
            content = normalized.get("content")
            if isinstance(content, str):
                normalized["content"] = _clip(content, max_chars)
        result.append(normalized)
    return result


def _first_content(messages: list[dict[str, Any]], roles: set[str]) -> str | None:
    for message in messages:
        if message.get("role") in roles and isinstance(message.get("content"), str):
            return str(message["content"])
    return None


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max(0, max_chars - 1)]}…"
