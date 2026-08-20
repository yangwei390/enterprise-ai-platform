from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionPhase(StrEnum):
    NEW = "NEW"
    WAITING_TOOL = "WAITING_TOOL"
    CONTINUE = "CONTINUE"
    READY_FOR_FINAL = "READY_FOR_FINAL"
    READY_FOR_CLARIFICATION = "READY_FOR_CLARIFICATION"
    FAILED = "FAILED"


class TransactionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    EXECUTING = "EXECUTING"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"


class ReferenceExpression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    ordinal: int | None = Field(default=None, ge=0)
    category_hint: str | None = None
    explicit_code: str | None = None
    explicit_name: str | None = None


class SemanticFrame(BaseModel):
    """LLM/规则只能描述用户表达，不能声明数据库实体或 Tool。"""

    model_config = ConfigDict(extra="forbid")

    intent: str
    slots: dict[str, Any] = Field(default_factory=dict)
    references: list[ReferenceExpression] = Field(default_factory=list)
    question: str | None = None
    continuation: bool = False
    requested_count: int | None = Field(default=None, ge=1, le=5)
    requires_manual_evidence: bool = False


class CandidateProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_code: str
    name: str
    category: str | None = None
    batch_id: str
    position: int = Field(ge=0)
    primary_manual_document_id: int | None = None


class ProductCandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    query: str
    category: str | None = None
    source_turn_id: str | None = Field(default=None, max_length=128)
    items: list[CandidateProduct] = Field(default_factory=list, max_length=20)


ProductQuestionPredicate = Literal[
    "bluetooth_connectivity",
    "charging",
    "compatibility",
    "price",
    "features",
    "buttons",
    "use_case",
    "usage",
    "warranty",
]


class ProductQuestionFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: ProductQuestionPredicate
    batch_id: str
    source_turn_id: str | None = None
    source_question: str | None = None


class PendingProductQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    keyword: str
    requested_count: int = Field(default=1, ge=1, le=5)
    filters: dict[str, Any] = Field(default_factory=dict)
    created_turn_id: str


class ProductContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_category: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    active_batch: ProductCandidateBatch | None = None
    active_product_code: str | None = None
    last_question: ProductQuestionFocus | None = None
    pending_query: PendingProductQuery | None = None


class OrderCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_ref: str = Field(min_length=1, max_length=128)
    display_label: str = Field(min_length=1, max_length=256)
    batch_id: str = Field(min_length=1, max_length=128)
    position: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_tool_result(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        order_ref = migrated.get("order_ref") or migrated.get("order_no")
        if order_ref is not None:
            migrated["order_ref"] = str(order_ref)
        migrated.setdefault("display_label", str(order_ref or ""))
        known = {"order_ref", "display_label", "batch_id", "position", "details"}
        details = migrated.get("details")
        normalized_details = dict(details) if isinstance(details, dict) else {}
        normalized_details.update(
            {key: item for key, item in migrated.items() if key not in known}
        )
        return {
            "order_ref": migrated.get("order_ref"),
            "display_label": migrated.get("display_label"),
            "batch_id": migrated.get("batch_id"),
            "position": migrated.get("position"),
            "details": normalized_details,
        }


class OrderCandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, max_length=128)
    query: str = Field(default="", max_length=2000)
    source_turn_id: str | None = Field(default=None, max_length=128)
    items: list[OrderCandidate] = Field(default_factory=list, max_length=20)


class OrderContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_batch: OrderCandidateBatch | None = None
    active_order_ref: str | None = Field(default=None, min_length=1, max_length=128)


DialogDomain = Literal[
    "general",
    "product",
    "manual",
    "order",
    "logistics",
    "after_sales",
    "handoff",
]


class DialogFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_domain: DialogDomain | None = None
    active_action: str | None = Field(default=None, max_length=128)
    active_batch_id: str | None = Field(default=None, max_length=128)
    last_successful_predicate: ProductQuestionPredicate | None = None
    source_turn_id: str | None = Field(default=None, max_length=128)


ClarificationKind = Literal[
    "missing_slot",
    "ambiguous_reference",
    "confirm_safe_query",
    "missing_evidence",
    "confirm_write",
]


class ResumeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=128)
    domain: DialogDomain
    safe_slots: dict[str, Any] = Field(default_factory=dict)


class PendingClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clarification_id: str = Field(min_length=1, max_length=128)
    kind: ClarificationKind
    domain: DialogDomain
    target_description: str = Field(min_length=1, max_length=500)
    missing_fields: list[str] = Field(default_factory=list, max_length=20)
    resume_action: ResumeAction | None = None
    attempts: int = Field(default=1, ge=1, le=2)
    created_turn_id: str | None = Field(default=None, max_length=128)
    last_asked_turn_id: str | None = Field(default=None, max_length=128)


class DialogueContextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class UnderstandingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_query: str
    recent_dialogue: list[DialogueContextMessage] = Field(
        default_factory=list,
        max_length=6,
    )
    active_product_category: str | None = None
    active_product_codes: list[str] = Field(default_factory=list, max_length=20)
    pending_product_query: PendingProductQuery | None = None
    pending_clarification: PendingClarification | None = None


class PendingAfterSales(BaseModel):
    model_config = ConfigDict(extra="allow")

    draft_id: str
    operation_id: str
    order_no: str
    customer_phone_last4: str
    status: str


class CustomerServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductContext = Field(default_factory=ProductContext)
    order: OrderContext = Field(default_factory=OrderContext)
    dialog_focus: DialogFocus = Field(default_factory=DialogFocus)
    pending_after_sales: PendingAfterSales | None = None
    pending_clarification: PendingClarification | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_state(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "product" not in migrated:
            filters = migrated.pop("filters", {})
            raw_batches = migrated.pop("candidate_batches", [])
            active_product_code = migrated.pop("active_product_code", None)
            latest_batch = (
                raw_batches[-1] if isinstance(raw_batches, list) and raw_batches else None
            )
            if isinstance(latest_batch, ProductCandidateBatch):
                latest_batch = latest_batch.model_dump(mode="python")
            category = filters.get("category") if isinstance(filters, dict) else None
            if category is None and isinstance(latest_batch, dict):
                category = latest_batch.get("category")
                if category is None:
                    items = latest_batch.get("items")
                    if isinstance(items, list) and items and isinstance(items[0], dict):
                        category = items[0].get("category")
            migrated["product"] = {
                "active_category": category,
                "filters": filters if isinstance(filters, dict) else {},
                "active_batch": latest_batch,
                "active_product_code": active_product_code,
            }

        legacy_order_candidates = migrated.pop("order_candidates", [])
        legacy_active_order_ref = migrated.pop("active_order_ref", None)
        if "order" not in migrated:
            batch_id = "legacy-order-batch"
            items = [
                {
                    **dict(item),
                    "batch_id": batch_id,
                    "position": index,
                }
                for index, item in enumerate(legacy_order_candidates)
                if isinstance(item, dict)
            ]
            migrated["order"] = {
                "active_batch": (
                    {
                        "batch_id": batch_id,
                        "query": "",
                        "source_turn_id": None,
                        "items": items,
                    }
                    if items
                    else None
                ),
                "active_order_ref": legacy_active_order_ref,
            }

        clarification_target = migrated.pop("clarification_target", None)
        clarification_rounds = migrated.pop("clarification_rounds", 0)
        if "pending_clarification" not in migrated and isinstance(
            clarification_target, str
        ) and clarification_target:
            attempts = clarification_rounds if isinstance(clarification_rounds, int) else 1
            migrated["pending_clarification"] = {
                "clarification_id": "legacy-clarification",
                "kind": "missing_slot",
                "domain": "general",
                "target_description": clarification_target,
                "missing_fields": [clarification_target],
                "attempts": min(2, max(1, attempts)),
            }
        return migrated


class SearchProductsCommand(BaseModel):
    kind: Literal["search_products"] = "search_products"
    keyword: str | None = None
    category: str | None = None
    product_code: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    page_size: int = Field(default=20, ge=1, le=100)
    knowledge_base_id: int | None = Field(default=None, ge=1)


class RecommendProductsCommand(BaseModel):
    kind: Literal["recommend_products"] = "recommend_products"
    filters: dict[str, Any] = Field(default_factory=dict)
    page_size: int = Field(default=1, ge=1, le=5)
    knowledge_base_id: int | None = Field(default=None, ge=1)


class CompareProductsCommand(BaseModel):
    kind: Literal["compare_products"] = "compare_products"
    product_codes: list[str] = Field(min_length=2, max_length=5)
    fields: list[str] = Field(default_factory=list)
    knowledge_base_id: int | None = Field(default=None, ge=1)


class KnowledgeSearchCommand(BaseModel):
    kind: Literal["knowledge_search"] = "knowledge_search"
    query: str
    knowledge_base_id: int = Field(ge=1)
    document_id: int = Field(ge=1)
    conversation_id: int | None = None
    memory_context: str | None = None


class QueryOrderCommand(BaseModel):
    kind: Literal["query_order"] = "query_order"
    order_ref: str | None = None


class QueryLogisticsCommand(BaseModel):
    kind: Literal["query_logistics"] = "query_logistics"
    order_ref: str


class CreateAfterSalesDraftCommand(BaseModel):
    kind: Literal["after_sales_draft"] = "after_sales_draft"
    action: Literal["draft"] = "draft"
    order_no: str
    customer_phone_last4: str
    issue_type: Literal["quality", "repair", "return", "exchange", "other"]
    issue_description: str = Field(min_length=5, max_length=1000)


class ConfirmAfterSalesCommand(BaseModel):
    kind: Literal["after_sales_confirm"] = "after_sales_confirm"
    action: Literal["confirm"] = "confirm"
    order_no: str
    customer_phone_last4: str
    draft_id: str
    operation_id: str
    confirmed: Literal[True] = True


class CreateHumanHandoffCommand(BaseModel):
    kind: Literal["human_handoff"] = "human_handoff"
    order_no: str
    customer_phone_last4: str
    reason: Literal["customer_request", "complaint", "tool_unavailable", "other"] = (
        "customer_request"
    )
    message: str = Field(min_length=2, max_length=1000)


CustomerServiceCommand = Annotated[
    SearchProductsCommand
    | RecommendProductsCommand
    | CompareProductsCommand
    | KnowledgeSearchCommand
    | QueryOrderCommand
    | QueryLogisticsCommand
    | CreateAfterSalesDraftCommand
    | ConfirmAfterSalesCommand
    | CreateHumanHandoffCommand,
    Field(discriminator="kind"),
]


class StatePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_before: CustomerServiceState
    proposed_state: CustomerServiceState
    proposed_patch: dict[str, Any] = Field(default_factory=dict)
    change_log: list[dict[str, Any]] = Field(default_factory=list)


class PendingTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    turn_id: str
    sequence: int = Field(ge=1, le=2)
    command: CustomerServiceCommand
    tool_call_id: str
    tool_name: str
    arguments_hash: str
    proposed_patch: dict[str, Any] = Field(default_factory=dict)
    expected_result_type: str
    status: TransactionStatus = TransactionStatus.PROPOSED


class GoalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_query: str
    intent: str | None = None
    semantic_frame: SemanticFrame | None = None
    resolved_entities: list[str] = Field(default_factory=list)


class CustomerServiceExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: ExecutionPhase = ExecutionPhase.NEW
    turn_id: str
    goal: GoalSnapshot | None = None
    pending_transaction: PendingTransaction | None = None
    verified_products: list[CandidateProduct] = Field(default_factory=list)
    verified_order_ref: str | None = None
    tool_count: int = Field(default=0, ge=0, le=2)
    failure_reason: str | None = None
