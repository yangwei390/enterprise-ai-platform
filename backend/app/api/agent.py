import asyncio
import json
from collections.abc import AsyncIterator, Awaitable
from typing import Any, cast

from backend.app.agents import (
    AgentRunRequest,
    AgentRuntimeFactory,
    AgentRuntimeRequest,
)
from backend.app.agents.catalog import AgentCatalog
from backend.app.agents.customer_service_contract import (
    customer_service_allowed_knowledge_base_ids,
)
from backend.app.agents.schemas import (
    AgentAssistantListResponse,
    AgentChatRequest,
    AgentChatResponseData,
    AgentStreamRequest,
)
from backend.app.agents.service import AgentService
from backend.app.agents.trace_builder import sanitize
from backend.app.config.settings import settings
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


def _trusted_customer_service_knowledge_scope(agent_id: str | None) -> frozenset[int]:
    return customer_service_allowed_knowledge_base_ids(
        agent_id=agent_id,
        configured_ids=settings.CUSTOMER_SERVICE_ALLOWED_KNOWLEDGE_BASE_IDS,
    )


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
        agent_id=request.agent_id,
        knowledge_base_id=request.knowledge_base_id,
        allowed_knowledge_base_ids=_trusted_customer_service_knowledge_scope(
            request.agent_id
        ),
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
            agent_id=request.agent_id,
            knowledge_base_id=request.knowledge_base_id,
            allowed_knowledge_base_ids=_trusted_customer_service_knowledge_scope(
                request.agent_id
            ),
            conversation_id=conversation_id,
            memory_context=request.memory_context,
            metadata=request.metadata,
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
                "trace_id": final_result.metadata.get("trace_id"),
                "runtime": final_result.metadata.get("runtime"),
                "agent_id": final_result.metadata.get("agent_id") or request.agent_id,
                "grounded_answer": final_result.metadata.get("grounded_answer"),
                "source_count": final_result.metadata.get("source_count"),
                "duration_ms": (
                    final_result.metadata.get("agent_trace", {})
                    .get("timing", {})
                    .get("total_duration_ms")
                )
                if isinstance(final_result.metadata.get("agent_trace"), dict)
                else None,
                "debug_trace": _customer_service_turn_debug(
                    request=request,
                    result=final_result,
                    answer=answer,
                ),
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


def _customer_service_turn_debug(
    *,
    request: AgentStreamRequest,
    result: AgentChatResponseData,
    answer: str,
) -> dict | None:
    if (
        not settings.CUSTOMER_SERVICE_TURN_DEBUG_ENABLED
        or request.agent_id != "customer_service_agent"
    ):
        return None
    customer_service = result.metadata.get("customer_service")
    trace = result.metadata.get("agent_trace")
    customer_service_data = customer_service if isinstance(customer_service, dict) else {}
    runtime_trace = trace if isinstance(trace, dict) else {}
    payload = {
        "input": {
            "query": request.query,
            "conversation_id": request.conversation_id,
            "knowledge_base_id": request.knowledge_base_id,
        },
        "steps": _customer_service_debug_steps(
            request=request,
            result=result,
            answer=answer,
            customer_service=customer_service_data,
            runtime_trace=runtime_trace,
        ),
        "customer_service": customer_service_data,
        "runtime_trace": runtime_trace,
        "tool_calls": result.tool_calls,
        "observations": result.observations,
        "output": {
            "action": result.action,
            "answer": answer,
        },
    }
    sanitized = sanitize(payload)
    return sanitized if isinstance(sanitized, dict) else {}


def _customer_service_debug_steps(
    *,
    request: AgentStreamRequest,
    result: AgentChatResponseData,
    answer: str,
    customer_service: dict,
    runtime_trace: dict,
) -> list[dict]:
    turn_debug = customer_service.get("turn_debug")
    turn_debug = turn_debug if isinstance(turn_debug, dict) else {}
    route = customer_service.get("route")
    route = route if isinstance(route, dict) else {}
    contextualized = customer_service.get("contextualized_request")
    contextualized = contextualized if isinstance(contextualized, dict) else {}
    payload = contextualized.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    dst_before = turn_debug.get("dst_before")
    dst_after = turn_debug.get("dst_after") or customer_service.get("dst")
    directive = turn_debug.get("fsm_directive") or customer_service.get("fsm_directive")
    evidence = runtime_trace.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}

    return [
        _debug_step(
            1,
            "输入预处理与前置路由",
            input_data={"query": request.query},
            output_data=turn_debug.get("pre_route"),
            executed=bool(turn_debug.get("pre_route")),
        ),
        _debug_step(
            2,
            "意图识别",
            input_data={"query": request.query, "pre_route": turn_debug.get("pre_route")},
            output_data={
                "route": route,
                "llm_classification": turn_debug.get("llm_classification"),
                "classification_failure_reason": turn_debug.get(
                    "classification_failure_reason"
                ),
            },
            executed=bool(route),
        ),
        _debug_step(
            3,
            "上下文化理解",
            input_data={
                "raw_query": request.query,
                "dst_before": dst_before,
            },
            output_data={
                "rewritten_query": contextualized.get("rewritten_query"),
                "domain": contextualized.get("domain"),
                "intent": contextualized.get("intent"),
                "target_references": route.get("target_references"),
                "target_product_codes": payload.get("target_product_codes"),
                "constraints": payload.get("constraints"),
                "order_action": payload.get("action"),
                "target_order_refs": payload.get("target_order_refs"),
            },
            executed=bool(contextualized),
        ),
        _debug_step(
            4,
            "生成 ContextualizedRequest",
            input_data={"route": route},
            output_data=contextualized,
            executed=bool(contextualized),
        ),
        _debug_step(
            5,
            "DST 与 FSM 状态更新",
            input_data={"dst_before": dst_before},
            output_data={
                "dst_after": dst_after,
                "fsm_directive": directive,
            },
            executed=dst_after is not None or directive is not None,
        ),
        _debug_step(
            6,
            "Tool 路由与参数组装",
            input_data={
                "intent": contextualized.get("intent") or route.get("intent"),
                "fsm_directive": directive,
            },
            output_data={"tool_calls": result.tool_calls},
            executed=bool(result.tool_calls),
            skipped_reason="本轮直接回答，没有选择 Tool",
        ),
        _debug_step(
            7,
            "业务 Tool 执行",
            input_data={"tool_calls": result.tool_calls},
            output_data={"observations": result.observations},
            executed=bool(result.observations),
            skipped_reason="本轮没有执行 Tool",
        ),
        _debug_step(
            8,
            "证据校验",
            input_data={
                "observations": result.observations,
                "retrieval": runtime_trace.get("retrieval"),
            },
            output_data={
                "evidence": evidence,
                "grounded_answer": result.metadata.get("grounded_answer"),
                "no_evidence": result.metadata.get("no_evidence"),
                "source_count": result.metadata.get("source_count"),
            },
            executed=bool(evidence)
            or result.metadata.get("grounded_answer") is not None
            or result.metadata.get("no_evidence") is not None,
            skipped_reason="本轮没有需要校验的外部证据",
        ),
        _debug_step(
            9,
            "最终回答",
            input_data={
                "action": result.action,
                "termination_reason": runtime_trace.get("final_answer"),
            },
            output_data={"answer": answer},
            executed=True,
        ),
    ]


def _debug_step(
    number: int,
    name: str,
    *,
    input_data: object,
    output_data: object,
    executed: bool,
    skipped_reason: str | None = None,
) -> dict:
    return {
        "step": number,
        "name": name,
        "status": "completed" if executed else "skipped",
        "input": input_data,
        "output": output_data,
        "reason": None if executed else skipped_reason or "本轮没有采集到该步骤结果",
    }


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/agents/run", response_model=ApiResponse)
def run_agent(request: AgentRunRequest) -> ApiResponse:
    result = AgentService().run(request)
    return success(data=result)
