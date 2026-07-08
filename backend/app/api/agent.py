from backend.app.agents import AgentRunRequest, AgentRuntime, AgentRuntimeRequest
from backend.app.agents.schemas import AgentChatRequest, AgentChatResponseData
from backend.app.agents.service import AgentService
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter

router = APIRouter()


@router.post("/agent/chat", response_model=ApiResponse)
def agent_chat(request: AgentChatRequest) -> ApiResponse:
    result = AgentRuntime().run(
        AgentRuntimeRequest(
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=request.conversation_id,
            memory_context=request.memory_context,
            metadata=request.metadata,
        )
    )
    return success(data=AgentChatResponseData.model_validate(result.model_dump()))


@router.post("/agents/run", response_model=ApiResponse)
def run_agent(request: AgentRunRequest) -> ApiResponse:
    result = AgentService().run(request)
    return success(data=result)
