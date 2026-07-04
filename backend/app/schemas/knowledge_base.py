from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str | None = None
    embedding_model: str | None = None
    vector_store: str = "qdrant"


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    embedding_model: str | None = None
    vector_store: str | None = None


class KnowledgeBaseResponse(BaseModel):
    id: int
    name: str
    description: str | None
    embedding_model: str | None
    vector_store: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseResponse]
    total: int
