from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalOperator = Literal["eq", "in", "contains", "prefix", "range"]
RetrievalIntent = Literal[
    "semantic",
    "lexical",
    "hybrid",
    "structured",
    "complex",
    "factual",
    "summary",
    "comparison",
    "multi_document",
    "open_query",
]
RetrievalStrategy = Literal["dense", "sparse", "hybrid", "structured_hybrid"]


class RetrievalConstraint(BaseModel):
    field: str
    operator: RetrievalOperator
    value: Any
    confidence: float = 1.0
    source: str = "rule"
    source_detail: str | None = None
    applied: bool = False
    rejected_reason: str | None = None


class ConstraintFieldDefinition(BaseModel):
    field: str
    value_type: Literal["int", "str", "list"]
    allowed_operators: list[RetrievalOperator]
    aliases: list[str] = Field(default_factory=list)
    enabled: bool = True


class QueryAnalysisResult(BaseModel):
    intent: RetrievalIntent
    constraints: list[RetrievalConstraint] = Field(default_factory=list)
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class RetrievalPlan(BaseModel):
    original_query: str
    rewritten_query: str
    intent: RetrievalIntent
    strategy: RetrievalStrategy
    document_ids: list[int] = Field(default_factory=list)
    constraints: list[RetrievalConstraint] = Field(default_factory=list)
    dense_enabled: bool = True
    sparse_enabled: bool = True
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    use_structure_filter: bool = False
    planner_source: str = "fast"
    fallback_used: bool = False
    fallback_reason: str | None = None
    metadata: dict = Field(default_factory=dict)
