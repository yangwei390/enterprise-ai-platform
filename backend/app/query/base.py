from typing import Literal

from pydantic import BaseModel, Field

RewriteType = Literal[
    "NONE",
    "NORMALIZATION",
    "EXPANSION",
    "DISAMBIGUATION",
    "KEYWORD_ENRICHMENT",
]


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    rewrite_reason: str = "no_rewrite_needed"
    rewrite_type: RewriteType = "NONE"
    rewrite_changed: bool = False
    changed: bool = False
    duration_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)
