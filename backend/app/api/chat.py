from backend.app.chat import ChatRequest, ChatResponse, ChatService
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter

router = APIRouter()


@router.post("/chat", response_model=ApiResponse)
def chat(request: ChatRequest) -> ApiResponse:
    service = ChatService()
    response = service.chat(request)
    return success(data=ChatResponse.model_validate(response.model_dump()))
