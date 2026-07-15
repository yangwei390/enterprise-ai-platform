from typing import Any, Literal

from pydantic import BaseModel, Field

QueryIntent = Literal[
    "factual",
    "summary",
    "structured",
    "comparison",
    "multi_document",
    "lexical",
    "open_query",
]


class QueryUnderstandingResult(BaseModel):
    original_query: str
    normalized_query: str
    intent: QueryIntent = "open_query"
    keywords: list[str] = Field(default_factory=list)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    document_hints: list[str] = Field(default_factory=list)
    structure_hints: list[dict[str, Any]] = Field(default_factory=list)
    temporal_constraints: list[dict[str, Any]] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    analyzer_source: str = "fast_rule_based"
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryUnderstandingTrace(BaseModel):
    enabled: bool
    intent: QueryIntent | None = None
    confidence: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    document_hints: list[str] = Field(default_factory=list)
    structure_hints: list[dict[str, Any]] = Field(default_factory=list)
    temporal_constraints: list[dict[str, Any]] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    analyzer_source: str | None = None
    duration_ms: float = 0.0
    failed: bool = False
    error: str | None = None
