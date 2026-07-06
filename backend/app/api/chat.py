import json
from collections.abc import Iterator

from backend.app.chat import ChatRequest, ChatResponse, ChatService
from backend.app.conversations import ConversationRepository, ConversationService
from backend.app.db.session import get_db
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
    response = service.chat(request)
    return StreamingResponse(
        _stream_chat_response(response),
        media_type="text/event-stream",
    )


def _stream_chat_response(response: ChatResponse) -> Iterator[str]:
    for content in response.answer:
        payload = json.dumps({"content": content}, ensure_ascii=False)
        yield f"event: delta\ndata: {payload}\n\n"

    done_payload = json.dumps(
        {
            "conversation_id": response.conversation_id,
            "message_id": response.message_id,
        },
        ensure_ascii=False,
    )
    yield f"event: done\ndata: {done_payload}\n\n"
