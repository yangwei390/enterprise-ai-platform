import asyncio
import json
from collections.abc import AsyncIterator, Awaitable
from typing import Any, cast

from backend.app.agents import (
    AgentRunRequest,
    AgentRuntime,
    AgentRuntimeFactory,
    AgentRuntimeRequest,
)
from backend.app.agents.catalog import AgentCatalog
from backend.app.agents.schemas import (
    AgentAssistantListResponse,
    AgentChatRequest,
    AgentChatResponseData,
    AgentStreamRequest,
)
from backend.app.agents.service import AgentService
from backend.app.conversations import ConversationRepository, ConversationService
from backend.app.db.session import get_db
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.conversation import ConversationCreate
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter()

V1_AGENT_RUNTIME_CLASS = AgentRuntime


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    repository = ConversationRepository(db)
    return ConversationService(repository)


@router.get("/agent/assistants", response_model=ApiResponse)
def list_agent_assistants() -> ApiResponse:
    assistants = AgentCatalog().list_assistants()
    return success(data=AgentAssistantListResponse(items=assistants, total=len(assistants)))


@router.get("/agent/assistants/{agent_id}", response_model=ApiResponse)
def get_agent_assistant(agent_id: str) -> ApiResponse:
    assistant = AgentCatalog().get_assistant(agent_id)
    if assistant is None:
        raise BusinessException(40404, "智能助手不存在")
    return success(data=assistant)


@router.post("/agent/chat", response_model=ApiResponse)
async def agent_chat(request: AgentChatRequest) -> ApiResponse:
    runtime = AgentRuntimeFactory.get_runtime()
    runtime_request = AgentRuntimeRequest(
        query=request.query,
        knowledge_base_id=request.knowledge_base_id,
        conversation_id=request.conversation_id,
        memory_context=request.memory_context,
        metadata=request.metadata,
    )
    arun = getattr(runtime, "arun", None)
    if callable(arun):
        result = await cast(Awaitable[Any], arun(runtime_request))
    else:
        result = await asyncio.to_thread(runtime.run, runtime_request)
    return success(data=AgentChatResponseData.model_validate(result.model_dump()))


@router.post("/agent/chat/stream")
async def agent_chat_stream(
    request: AgentStreamRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_agent_events(request, conversation_service),
        media_type="text/event-stream",
    )


async def _stream_agent_events(
    request: AgentStreamRequest,
    conversation_service: ConversationService,
) -> AsyncIterator[str]:
    try:
        conversation_id = _prepare_agent_conversation(request, conversation_service)
        if request.conversation_id is None:
            request = request.model_copy(update={"conversation_id": conversation_id})
        conversation_service.add_user_message(
            conversation_id=conversation_id,
            content=request.query,
            metadata={
                "agent_id": request.agent_id,
                "workspace": "agent",
            },
        )
        yield _sse(
            "message_start",
            {
                "conversation_id": conversation_id,
                "role": "assistant",
                "agent_id": request.agent_id,
            },
        )

        runtime = AgentRuntimeFactory.get_runtime()
        runtime_request = AgentRuntimeRequest(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=conversation_id,
            memory_context=request.memory_context,
            metadata={**request.metadata, "agent_id": request.agent_id},
        )
        final_result = None
        answer_parts: list[str] = []
        astream_events = getattr(runtime, "astream_events", None)
        if callable(astream_events):
            event_stream = cast(AsyncIterator[dict], astream_events(runtime_request))
            async for item in event_stream:
                event = item.get("event")
                data = item.get("data", {})
                if event == "status":
                    yield _sse("status", data)
                elif event == "answer_delta":
                    delta = str(data.get("delta") or "")
                    if delta:
                        answer_parts.append(delta)
                        yield _sse("answer_delta", {"delta": delta})
                elif event == "result":
                    final_result = AgentChatResponseData.model_validate(
                        data["result"]
                    )
        else:
            yield _sse(
                "status",
                {"status": "processing", "message": "正在处理任务"},
            )
            arun = getattr(runtime, "arun", None)
            if callable(arun):
                raw_result = await cast(Awaitable[Any], arun(runtime_request))
            else:
                raw_result = await asyncio.to_thread(runtime.run, runtime_request)
            final_result = AgentChatResponseData.model_validate(raw_result.model_dump())
            yield _sse(
                "status",
                {"status": "answering", "message": "正在整理答案"},
            )

        if final_result is None:
            raise RuntimeError("agent stream ended without result")

        answer = final_result.answer or "智能助手没有生成可展示的回答。"
        streamed_answer = "".join(answer_parts)
        if streamed_answer:
            answer = streamed_answer
        elif answer:
            yield _sse("answer_delta", {"delta": answer})
        citations = final_result.citations
        sources = final_result.sources
        if citations or sources:
            yield _sse("citations", {"citations": citations, "sources": sources})

        assistant_message = conversation_service.add_assistant_message(
            conversation_id=conversation_id,
            content=answer,
            metadata={
                "agent_id": request.agent_id,
                "workspace": "agent",
                "action": final_result.action,
                "sources": sources,
                "citations": citations,
                "agent_loop": final_result.metadata.get("agent_loop", {}),
                "async_runtime": final_result.metadata.get("async_runtime", {}),
            },
        )
        yield _sse(
            "completed",
            {
                "conversation_id": conversation_id,
                "message_id": assistant_message.id,
                "answer": answer,
                "citations": citations,
                "sources": sources,
                "status": "completed",
            },
        )
    except asyncio.CancelledError:
        logger.info("Agent stream cancelled by client")
        raise
    except Exception:
        logger.exception("Streaming agent failed")
        yield _sse(
            "error",
            {"message": "智能助手执行任务时发生错误，请稍后重试。"},
        )


def _prepare_agent_conversation(
    request: AgentStreamRequest,
    conversation_service: ConversationService,
) -> int:
    if request.conversation_id is not None:
        conversation_service.get_conversation(request.conversation_id)
        return request.conversation_id
    conversation = conversation_service.create_conversation(
        ConversationCreate(
            title=f"Agent: {request.query[:24]}",
            knowledge_base_id=request.knowledge_base_id,
        )
    )
    return conversation.id


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/agents/run", response_model=ApiResponse)
def run_agent(request: AgentRunRequest) -> ApiResponse:
    result = AgentService().run(request)
    return success(data=result)
