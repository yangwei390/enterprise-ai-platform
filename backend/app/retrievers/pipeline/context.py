from dataclasses import dataclass, field

from backend.app.context import ContextChunk
from backend.app.rerankers import RerankedChunk
from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.metadata_filter import AutoMetadataFilterResult
from backend.app.retrievers.planning import RetrievalPlan


@dataclass
class RetrieverPipelineContext:
    query: str
    original_query: str | None = None
    rewritten_query: str | None = None
    knowledge_base_id: int | None = None
    top_k: int = 5
    score_threshold: float | None = None
    metadata_filter: dict | None = None
    dense_chunks: list[RetrievedChunk] = field(default_factory=list)
    sparse_chunks: list[RetrievedChunk] = field(default_factory=list)
    fused_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranked_chunks: list[RerankedChunk] = field(default_factory=list)
    context_chunks: list[ContextChunk] = field(default_factory=list)
    context_text: str = ""
    metadata: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    auto_filter_result: AutoMetadataFilterResult | None = None
    retrieval_plan: RetrievalPlan | None = None

    @property
    def active_query(self) -> str:
        return self.rewritten_query or self.query

    def add_error(self, step: str, error: Exception) -> None:
        self.errors.append(
            {
                "step": step,
                "error": str(error),
            }
        )
