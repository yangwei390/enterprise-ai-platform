from backend.app.agents import AgentRunRequest, AgentService
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter

router = APIRouter()


@router.post("/agents/run", response_model=ApiResponse)
def run_agent(request: AgentRunRequest) -> ApiResponse:
    result = AgentService().run(request)
    return success(data=result)
