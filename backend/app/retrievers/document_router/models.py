from typing import Literal

from pydantic import BaseModel, Field

RouteType = Literal["DOCUMENT", "KNOWLEDGE_BASE", "MULTI_DOCUMENT", "UNKNOWN"]


class RoutingResult(BaseModel):
    route_type: RouteType
    target_document_ids: list[int] = Field(default_factory=list)
    candidate_document_ids: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = "unknown"
    metadata: dict = Field(default_factory=dict)
