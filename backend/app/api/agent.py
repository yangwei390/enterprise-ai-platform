import asyncio
from collections.abc import Awaitable
from typing import Any, cast

from backend.app.agents import (
    AgentRunRequest,
    AgentRuntime,
    AgentRuntimeFactory,
    AgentRuntimeRequest,
)
from backend.app.agents.schemas import AgentChatRequest, AgentChatResponseData
from backend.app.agents.service import AgentService
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter

router = APIRouter()

V1_AGENT_RUNTIME_CLASS = AgentRuntime


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


@router.post("/agents/run", response_model=ApiResponse)
def run_agent(request: AgentRunRequest) -> ApiResponse:
    result = AgentService().run(request)
    return success(data=result)
