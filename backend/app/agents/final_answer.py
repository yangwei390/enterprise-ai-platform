import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from backend.app.llms import LLMFactory, LLMMessage, LLMRequest

DeltaCallback = Callable[[str], None | Awaitable[None]]


def build_final_answer_request(
    *,
    query: str,
    observations: list[dict],
    knowledge: dict | None = None,
    fallback_answer: str | None = None,
) -> LLMRequest:
    context_parts = []
    if knowledge:
        context_parts.append(
            "知识库结果：\n"
            + json.dumps(
                {
                    "answer": knowledge.get("answer"),
                    "citations": knowledge.get("citations", []),
                },
                ensure_ascii=False,
                default=str,
            )
        )
    if observations:
        context_parts.append(
            "执行结果：\n"
            + json.dumps(
                [_summarize_observation(observation) for observation in observations],
                ensure_ascii=False,
                default=str,
            )
        )
    if fallback_answer:
        context_parts.append(f"已有结果摘要：\n{fallback_answer}")

    context_text = "\n\n".join(part for part in context_parts if part.strip())
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "你是企业级 AI 助手。请只面向最终用户输出答案，"
                    "不要暴露工具调用、协议、内部状态或调试信息。"
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"用户问题：{query}\n\n"
                    f"{context_text}\n\n"
                    "请基于以上信息生成最终回答。"
                ),
            ),
        ],
        temperature=0,
        metadata={"agent_final_answer_stream": True},
    )


async def stream_final_answer(
    request: LLMRequest,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    llm = LLMFactory.get_llm()

    def run_stream() -> None:
        try:
            for delta in llm.stream(request):
                if delta:
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
        except BaseException as exc:  # noqa: BLE001 - propagated through async queue
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    stream_task = asyncio.create_task(asyncio.to_thread(run_stream))
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        await stream_task


async def collect_streaming_answer(
    request: LLMRequest,
    *,
    on_delta: DeltaCallback | None = None,
) -> str:
    parts: list[str] = []
    async for delta in stream_final_answer(request):
        parts.append(delta)
        if on_delta is not None:
            result = on_delta(delta)
            if asyncio.iscoroutine(result):
                await result
    return "".join(parts)


def _summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": observation.get("success"),
        "content": observation.get("content"),
        "result": observation.get("result")
        if isinstance(observation.get("result"), str | int | float | bool)
        else observation.get("raw_result"),
        "error": observation.get("error"),
    }
