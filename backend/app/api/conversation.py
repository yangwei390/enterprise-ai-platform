from backend.app.conversations import ConversationRepository, ConversationService
from backend.app.db.session import get_db
from backend.app.models import Message
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.conversation import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
)
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter()


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    repository = ConversationRepository(db)
    return ConversationService(repository)


def to_message_response(message: Message) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        metadata=message.message_metadata,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


@router.post("/conversations", response_model=ApiResponse)
def create_conversation(
    data: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
) -> ApiResponse:
    conversation = service.create_conversation(data)
    return success(data=ConversationResponse.model_validate(conversation))


@router.get("/conversations", response_model=ApiResponse)
def list_conversations(
    service: ConversationService = Depends(get_conversation_service),
) -> ApiResponse:
    conversations = service.list_conversations()
    items = [ConversationResponse.model_validate(item) for item in conversations]
    return success(data=ConversationListResponse(items=items, total=len(items)))


@router.get("/conversations/{id}", response_model=ApiResponse)
def get_conversation(
    id: int,
    service: ConversationService = Depends(get_conversation_service),
) -> ApiResponse:
    conversation = service.get_conversation(id)
    return success(data=ConversationResponse.model_validate(conversation))


@router.delete("/conversations/{id}", response_model=ApiResponse)
def delete_conversation(
    id: int,
    service: ConversationService = Depends(get_conversation_service),
) -> ApiResponse:
    service.delete_conversation(id)
    return success(data={"deleted": True})


@router.get("/conversations/{id}/messages", response_model=ApiResponse)
def list_messages(
    id: int,
    service: ConversationService = Depends(get_conversation_service),
) -> ApiResponse:
    messages = service.list_messages(id)
    return success(data=[to_message_response(message) for message in messages])
