from typing import Literal

from backend.app.retrievers.document_router import RouteType
from pydantic import BaseModel, Field

RetrievalStrategyType = Literal["DOCUMENT", "MULTI_DOCUMENT", "GLOBAL"]


class RetrievalStrategy(BaseModel):
    strategy: RetrievalStrategyType
    route_type: RouteType
    document_ids: list[int] = Field(default_factory=list)
    per_document_budget: int | None = None
    global_budget: int
    fallback: bool = False
    metadata: dict = Field(default_factory=dict)
