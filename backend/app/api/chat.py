import json
from collections.abc import Iterator

from backend.app.chat import ChatRequest, ChatResponse, ChatService
from backend.app.conversations import ConversationRepository, ConversationService
from backend.app.db.session import get_db
from backend.app.logger import logger
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter()


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    repository = ConversationRepository(db)
    return ConversationService(repository)


def get_chat_service(
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ChatService:
    return ChatService(conversation_service)


@router.post("/chat", response_model=ApiResponse)
def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ApiResponse:
    response = service.chat(request)
    return success(data=ChatResponse.model_validate(response.model_dump()))


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat_events(service, request),
        media_type="text/event-stream",
    )


def _stream_chat_events(service: ChatService, request: ChatRequest) -> Iterator[str]:
    try:
        for item in service.stream_chat_events(request):
            event = item.get("event", "message")
            payload = json.dumps(item.get("data", {}), ensure_ascii=False, default=str)
            yield f"event: {event}\ndata: {payload}\n\n"
    except GeneratorExit:
        raise
    except Exception:
        logger.exception("Streaming chat failed")
        payload = json.dumps(
            {"message": "生成回答时发生错误，请稍后重试。"},
            ensure_ascii=False,
        )
        yield f"event: error\ndata: {payload}\n\n"
