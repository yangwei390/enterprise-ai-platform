from pydantic import BaseModel, Field


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    changed: bool
    metadata: dict = Field(default_factory=dict)
