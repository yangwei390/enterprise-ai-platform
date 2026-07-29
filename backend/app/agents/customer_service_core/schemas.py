from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserDecision(StrEnum):
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    AMBIGUOUS = "AMBIGUOUS"
    OTHER = "OTHER"
    UNSAFE_INJECTION = "UNSAFE_INJECTION"


class CustomerServiceIntent(StrEnum):
    GREETING = "greeting"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PRODUCT_SEARCH = "product_search"
    PRODUCT_REALTIME_FACT = "product_realtime_fact"
    PRODUCT_DOCUMENT_FACT = "product_document_fact"
    PRODUCT_COMPARISON = "product_comparison"
    POLICY_QUESTION = "policy_question"
    ORDER_QUERY = "order_query"
    LOGISTICS_QUERY = "logistics_query"
    AFTER_SALES = "after_sales"
    HUMAN_HANDOFF = "human_handoff"
    OUT_OF_SCOPE = "out_of_scope"
    OTHER = "other"


class CustomerServiceSource(StrEnum):
    PRODUCT_CATALOG = "product_catalog"
    PRIMARY_MANUAL = "primary_manual"
    POLICY_KNOWLEDGE = "policy_knowledge"
    ORDER_SERVICE = "order_service"
    AFTER_SALES_WORKFLOW = "after_sales_workflow"
    HUMAN_HANDOFF = "human_handoff"
    PLANNER = "planner"


class CustomerServiceDomain(StrEnum):
    PRODUCT = "product"
    ORDER = "order"
    LOGISTICS = "logistics"
    AFTER_SALES = "after_sales"
    KNOWLEDGE = "knowledge"
    HUMAN_HANDOFF = "human_handoff"
    GENERAL = "general"


class CustomerServiceIntentMode(StrEnum):
    RULE_ONLY = "rule_only"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"


class CustomerServiceRecognitionSource(StrEnum):
    RULES = "rules"
    LLM = "llm"
    FALLBACK = "fallback"


class ProductRequestConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str | None = None
    brand: str | None = None
    category: str | None = None
    model: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    required_features: list[str] = Field(default_factory=list, max_length=10)
    preferred_features: list[str] = Field(default_factory=list, max_length=10)
    required_use_cases: list[str] = Field(default_factory=list, max_length=10)
    preferred_use_cases: list[str] = Field(default_factory=list, max_length=10)


class SlotOperation(StrEnum):
    SET = "SET"
    KEEP = "KEEP"
    REMOVE = "REMOVE"


class SlotUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: SlotOperation = SlotOperation.KEEP
    value: Any = None

    @model_validator(mode="after")
    def validate_operation(self) -> SlotUpdate:
        if self.op == SlotOperation.SET and self.value in (None, "", []):
            raise ValueError("SET 必须携带非空 value")
        if self.op != SlotOperation.SET and self.value is not None:
            raise ValueError("KEEP/REMOVE 不允许携带 value")
        return self


class ProductConstraintOperations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: SlotUpdate = Field(default_factory=SlotUpdate)
    brand: SlotUpdate = Field(default_factory=SlotUpdate)
    category: SlotUpdate = Field(default_factory=SlotUpdate)
    model: SlotUpdate = Field(default_factory=SlotUpdate)
    price_min: SlotUpdate = Field(default_factory=SlotUpdate)
    price_max: SlotUpdate = Field(default_factory=SlotUpdate)
    required_features: SlotUpdate = Field(default_factory=SlotUpdate)
    preferred_features: SlotUpdate = Field(default_factory=SlotUpdate)
    required_use_cases: SlotUpdate = Field(default_factory=SlotUpdate)
    preferred_use_cases: SlotUpdate = Field(default_factory=SlotUpdate)


class TargetCardinality(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    ALL = "all"
    NONE = "none"


class TargetResolutionSource(StrEnum):
    EXPLICIT = "explicit"
    ORDINAL = "ordinal"
    ACTIVE = "active"
    SINGLE_CANDIDATE = "single_candidate"
    ALL_CANDIDATES = "all_candidates"
    UNRESOLVED = "unresolved"


class TargetResolutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cardinality: TargetCardinality
    allow_active: bool = False
    allow_single_candidate: bool = False


class TargetResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved_ids: list[str] = Field(default_factory=list, max_length=5)
    source: TargetResolutionSource = TargetResolutionSource.UNRESOLVED
    clarification_required: bool = False
    out_of_range: bool = False


class ProductPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_product_codes: list[str] = Field(default_factory=list, max_length=5)
    attributes: list[str] = Field(default_factory=list, max_length=5)
    recommendation_count: int | None = Field(default=None, ge=1, le=100)
    resolution_source: TargetResolutionSource = TargetResolutionSource.UNRESOLVED
    constraints: ProductRequestConstraints = Field(default_factory=ProductRequestConstraints)
    constraint_operations: ProductConstraintOperations = Field(
        default_factory=ProductConstraintOperations
    )


class OrderAction(StrEnum):
    LIST = "list"
    COUNT = "count"
    DETAIL = "detail"
    LOGISTICS = "logistics"
    COMPARE = "compare"


class OrderScope(StrEnum):
    ALL = "all"
    SELECTED = "selected"


class OrderPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: OrderAction
    scope: OrderScope
    target_order_refs: list[str] = Field(default_factory=list, max_length=5)
    resolution_source: TargetResolutionSource = TargetResolutionSource.UNRESOLVED
    clarification_required: bool = False
    clarification_question: str | None = None


class KnowledgePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    product_ref: str | None = None
    document_id: int | None = Field(default=None, ge=1)


class AfterSalesPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_ref: str | None = None
    customer_phone_last4: str | None = Field(
        default=None,
        pattern=r"^\d{4}$",
    )
    issue_type: str | None = None
    reason: str | None = None
    confirmed: bool = False


class HandoffPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_ref: str | None = None
    customer_phone_last4: str | None = Field(
        default=None,
        pattern=r"^\d{4}$",
    )
    reason: str | None = None
    message: str | None = Field(default=None, max_length=500)


class GeneralPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str | None = None


class CustomerServiceIntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: CustomerServiceIntent
    domain: CustomerServiceDomain | None = None
    action: str | None = Field(default=None, max_length=64)
    confidence: float = Field(ge=0, le=1)
    target_references: list[str] = Field(default_factory=list, max_length=5)
    attributes: list[str] = Field(default_factory=list, max_length=5)
    recommendation_count: int | None = Field(default=None, ge=1, le=5)
    rewritten_query: str | None = None
    constraint_operations: ProductConstraintOperations = Field(
        default_factory=ProductConstraintOperations
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_constraints(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "constraints" not in value:
            return value
        migrated = dict(value)
        constraints = migrated.pop("constraints")
        if "constraint_operations" not in migrated and isinstance(constraints, dict):
            migrated["constraint_operations"] = {
                key: {"op": "SET", "value": item}
                for key, item in constraints.items()
                if item not in (None, "", [])
            }
        return migrated


_INTENT_DOMAIN = {
    CustomerServiceIntent.PRODUCT_RECOMMENDATION: CustomerServiceDomain.PRODUCT,
    CustomerServiceIntent.PRODUCT_SEARCH: CustomerServiceDomain.PRODUCT,
    CustomerServiceIntent.PRODUCT_REALTIME_FACT: CustomerServiceDomain.PRODUCT,
    CustomerServiceIntent.PRODUCT_DOCUMENT_FACT: CustomerServiceDomain.PRODUCT,
    CustomerServiceIntent.PRODUCT_COMPARISON: CustomerServiceDomain.PRODUCT,
    CustomerServiceIntent.ORDER_QUERY: CustomerServiceDomain.ORDER,
    CustomerServiceIntent.LOGISTICS_QUERY: CustomerServiceDomain.LOGISTICS,
    CustomerServiceIntent.AFTER_SALES: CustomerServiceDomain.AFTER_SALES,
    CustomerServiceIntent.POLICY_QUESTION: CustomerServiceDomain.KNOWLEDGE,
    CustomerServiceIntent.HUMAN_HANDOFF: CustomerServiceDomain.HUMAN_HANDOFF,
    CustomerServiceIntent.GREETING: CustomerServiceDomain.GENERAL,
    CustomerServiceIntent.OUT_OF_SCOPE: CustomerServiceDomain.GENERAL,
    CustomerServiceIntent.OTHER: CustomerServiceDomain.GENERAL,
}

_INTENT_ACTION = {
    CustomerServiceIntent.GREETING: "greet",
    CustomerServiceIntent.PRODUCT_RECOMMENDATION: "recommend_products",
    CustomerServiceIntent.PRODUCT_SEARCH: "search_products",
    CustomerServiceIntent.PRODUCT_REALTIME_FACT: "query_product_fact",
    CustomerServiceIntent.PRODUCT_DOCUMENT_FACT: "query_product_document",
    CustomerServiceIntent.PRODUCT_COMPARISON: "compare_products",
    CustomerServiceIntent.POLICY_QUESTION: "query_policy",
    CustomerServiceIntent.ORDER_QUERY: "query_order",
    CustomerServiceIntent.LOGISTICS_QUERY: "query_logistics",
    CustomerServiceIntent.AFTER_SALES: "after_sales",
    CustomerServiceIntent.HUMAN_HANDOFF: "human_handoff",
    CustomerServiceIntent.OUT_OF_SCOPE: "out_of_scope",
    CustomerServiceIntent.OTHER: "clarify",
}


def domain_for_intent(intent: CustomerServiceIntent) -> CustomerServiceDomain:
    return _INTENT_DOMAIN[intent]


def action_for_intent(intent: CustomerServiceIntent) -> str:
    return _INTENT_ACTION[intent]


def source_for_intent(intent: CustomerServiceIntent) -> CustomerServiceSource:
    if intent == CustomerServiceIntent.PRODUCT_DOCUMENT_FACT:
        return CustomerServiceSource.PRIMARY_MANUAL
    if intent == CustomerServiceIntent.POLICY_QUESTION:
        return CustomerServiceSource.POLICY_KNOWLEDGE
    if intent in {
        CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        CustomerServiceIntent.PRODUCT_SEARCH,
        CustomerServiceIntent.PRODUCT_REALTIME_FACT,
        CustomerServiceIntent.PRODUCT_COMPARISON,
    }:
        return CustomerServiceSource.PRODUCT_CATALOG
    if intent in {
        CustomerServiceIntent.ORDER_QUERY,
        CustomerServiceIntent.LOGISTICS_QUERY,
    }:
        return CustomerServiceSource.ORDER_SERVICE
    if intent == CustomerServiceIntent.AFTER_SALES:
        return CustomerServiceSource.AFTER_SALES_WORKFLOW
    if intent == CustomerServiceIntent.HUMAN_HANDOFF:
        return CustomerServiceSource.HUMAN_HANDOFF
    return CustomerServiceSource.PLANNER


class ContextualizedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_query: str
    rewritten_query: str
    domain: CustomerServiceDomain
    intent: CustomerServiceIntent
    action: str
    source: CustomerServiceSource
    intent_mode: CustomerServiceIntentMode = CustomerServiceIntentMode.HYBRID
    recognition_source: CustomerServiceRecognitionSource = CustomerServiceRecognitionSource.RULES
    target_references: list[str] = Field(default_factory=list, max_length=5)
    payload: (
        ProductPayload
        | OrderPayload
        | KnowledgePayload
        | AfterSalesPayload
        | HandoffPayload
        | GeneralPayload
        | None
    ) = None
    confidence: float = Field(default=1, ge=0, le=1)
    clarification_required: bool = False
    clarification_question: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_product_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        try:
            intent = CustomerServiceIntent(migrated.get("intent"))
        except ValueError:
            return value
        migrated.setdefault("domain", domain_for_intent(intent))
        migrated.setdefault("action", action_for_intent(intent))
        legacy_fields = {
            "target_product_codes",
            "attributes",
            "recommendation_count",
            "constraints",
        }
        if (
            migrated["domain"] == CustomerServiceDomain.PRODUCT
            and "payload" not in migrated
            and any(field in migrated for field in legacy_fields)
        ):
            migrated["payload"] = {
                field: migrated.get(field) for field in legacy_fields if field in migrated
            }
        for field in legacy_fields:
            migrated.pop(field, None)
        return migrated


class DialogStatus(StrEnum):
    IDLE = "idle"
    UNDERSTANDING = "understanding"
    COLLECTING_SLOTS = "collecting_slots"
    RESOLVING_TARGET = "resolving_target"
    READY_TO_EXECUTE = "ready_to_execute"
    EXECUTING = "executing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ANSWERING = "answering"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class SlotValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any = None
    source: str
    confidence: float = Field(default=1, ge=0, le=1)
    validated: bool = False


class DialogTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: CustomerServiceDomain
    ref: str
    display_name: str | None = None


class CandidateRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    ref: str
    display_name: str | None = None
    position: int = Field(ge=1)


class DomainState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateRef] = Field(default_factory=list, max_length=100)
    active_ref: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    seen_refs: list[str] = Field(default_factory=list, max_length=100)


class TaskFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: CustomerServiceDomain
    intent: CustomerServiceIntent
    status: DialogStatus
    slots: dict[str, SlotValue] = Field(default_factory=dict)


class PendingConfirmation(BaseModel):
    model_config = ConfigDict(extra="allow")

    operation_id: str
    action: str
    status: str


class ToolSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    status: str
    result_refs: list[str] = Field(default_factory=list)


class ConversationDST(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    revision: int = Field(default=0, ge=0)
    active_domain: CustomerServiceDomain = CustomerServiceDomain.GENERAL
    active_intent: CustomerServiceIntent = CustomerServiceIntent.OTHER
    status: DialogStatus = DialogStatus.IDLE
    raw_query: str = ""
    rewritten_query: str = ""
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    required_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    active_target: DialogTarget | None = None
    domains: dict[CustomerServiceDomain, DomainState] = Field(default_factory=dict)
    stack: list[TaskFrame] = Field(default_factory=list, max_length=10)
    pending_confirmation: PendingConfirmation | None = None
    last_tool: ToolSnapshot | None = None
    suppressed_slots: dict[str, str] = Field(default_factory=dict)
    slot_change_log: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    error: str | None = None
