from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    document_metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class EmbeddingPreview(BaseModel):
    chunk_index: int
    document_id: int | None
    knowledge_base_id: int | None
    dimension: int
    preview: list[float]


class DocumentParseResponse(BaseModel):
    document_id: int
    text_length: int
    preview: str
    page_count: int | None
    metadata: dict
    original_length: int
    cleaned_length: int
    cleaner_metadata: dict
    chunk_strategy: str
    chunk_size: int | None
    chunk_overlap: int | None
    total_chunks: int
    chunks_preview: list[dict]
    embedding_model: str | None
    embedding_dimension: int | None
    total_embeddings: int
    embeddings_preview: list[EmbeddingPreview]
    vector_collection: str | None
    vector_total_records: int | None
    vector_ids_preview: list[str]
