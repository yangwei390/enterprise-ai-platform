from pydantic import BaseModel, Field


class RagTraceChunk(BaseModel):
    id: str | None
    document_id: int | None
    knowledge_base_id: int | None
    chunk_index: int | None
    source: str | None
    text_preview: str
    score: float | None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RagTraceResult(BaseModel):
    query: str
    rewritten_query: str | None = None
    knowledge_base_id: int | None = None
    retriever_mode: str | None = None
    dense_chunks: list[RagTraceChunk] = Field(default_factory=list)
    sparse_chunks: list[RagTraceChunk] = Field(default_factory=list)
    fused_chunks: list[RagTraceChunk] = Field(default_factory=list)
    reranked_chunks: list[RagTraceChunk] = Field(default_factory=list)
    context_chunks: list[RagTraceChunk] = Field(default_factory=list)
    context_text_preview: str | None = None
    metadata: dict = Field(default_factory=dict)
