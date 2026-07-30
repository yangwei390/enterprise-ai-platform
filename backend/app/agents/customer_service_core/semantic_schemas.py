from __future__ import annotations

from enum import StrEnum
from typing import Any

from backend.app.agents.customer_service_core.schemas import (
    CustomerServiceDomain,
    CustomerServiceIntent,
    OrderAction,
    ProductConstraintOperations,
)
from pydantic import BaseModel, ConfigDict, Field


class FieldSource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class FieldProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    source: FieldSource


class TargetSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int | None = Field(default=None, ge=-1)
    ordinals: list[int] = Field(default_factory=list, max_length=5)
    explicit_ref: str | None = None
    explicit_refs: list[str] = Field(default_factory=list, max_length=5)
    explicit_ref_source: FieldSource | None = None
    explicit_name: str | None = None
    category_hint: str | None = None
    reference_text: str | None = None


class ProductSemanticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_ops: ProductConstraintOperations = Field(default_factory=ProductConstraintOperations)
    attributes: list[str] = Field(default_factory=list, max_length=5)
    recommendation_count: int | None = Field(default=None, ge=1, le=100)


class OrderSemanticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: OrderAction | None = None
    explicit_order_ref: str | None = None
    explicit_phone_last4: str | None = Field(default=None, pattern=r"^\d{4}$")


class AfterSalesSemanticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_type: str | None = None
    order_ref: str | None = None
    confirmed: bool = False


class KnowledgeSemanticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = None


class HandoffSemanticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_ref: str | None = None
    reason: str | None = None


SemanticPayload = (
    ProductSemanticPayload
    | OrderSemanticPayload
    | AfterSalesSemanticPayload
    | KnowledgeSemanticPayload
    | HandoffSemanticPayload
)


class SemanticParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_proposal: FieldProposal | None = None
    domain_proposal: FieldProposal | None = None
    target_semantics: TargetSemantics | None = None
    payload_proposal: SemanticPayload | None = None
    gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    needs_llm: bool = False

    @property
    def intent(self) -> CustomerServiceIntent | None:
        if self.intent_proposal is None:
            return None
        try:
            return CustomerServiceIntent(self.intent_proposal.value)
        except ValueError:
            return None

    @property
    def domain(self) -> CustomerServiceDomain | None:
        if self.domain_proposal is None:
            return None
        try:
            return CustomerServiceDomain(self.domain_proposal.value)
        except ValueError:
            return None
