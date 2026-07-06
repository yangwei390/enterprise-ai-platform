from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationCreate(BaseModel):
    title: str | None = None
    knowledge_base_id: int | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    metadata: dict | None = None
    created_at: datetime
    updated_at: datetime


class ConversationResponse(BaseModel):
    id: int
    title: str | None
    knowledge_base_id: int | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
