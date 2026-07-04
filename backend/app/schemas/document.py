from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentCreate(BaseModel):
    knowledge_base_id: int | None
    filename: str
    file_size: int = 0


class DocumentUpdate(BaseModel):
    status: str | None = None
    chunk_count: int | None = None


class DocumentResponse(BaseModel):
    id: int
    knowledge_base_id: int
    filename: str
    file_size: int
    status: str
    chunk_count: int
    original_filename: str | None
    storage_path: str | None
    mime_type: str | None
    file_hash: str | None
    parse_status: str
    parse_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class DocumentParseResponse(BaseModel):
    document_id: int
    text_length: int
    preview: str
    page_count: int | None
    metadata: dict
