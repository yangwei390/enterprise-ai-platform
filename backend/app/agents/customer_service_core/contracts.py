from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    items: list[CandidateProduct] = Field(default_factory=list, max_length=20)


class PendingAfterSales(BaseModel):
    model_config = ConfigDict(extra="allow")

    draft_id: str
    operation_id: str
    order_no: str
    customer_phone_last4: str
    status: str


class CustomerServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: dict[str, Any] = Field(default_factory=dict)
    candidate_batches: list[ProductCandidateBatch] = Field(default_factory=list, max_length=8)
    active_product_code: str | None = None
    order_candidates: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    active_order_ref: str | None = None
    pending_after_sales: PendingAfterSales | None = None
    clarification_target: str | None = None
    clarification_rounds: int = Field(default=0, ge=0, le=2)


class SearchProductsCommand(BaseModel):
    kind: Literal["search_products"] = "search_products"
    keyword: str | None = None
    category: str | None = None
    product_code: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    page_size: int = Field(default=20, ge=1, le=100)


class RecommendProductsCommand(BaseModel):
    kind: Literal["recommend_products"] = "recommend_products"
    filters: dict[str, Any] = Field(default_factory=dict)
    page_size: int = Field(default=3, ge=1, le=5)


class CompareProductsCommand(BaseModel):
    kind: Literal["compare_products"] = "compare_products"
    product_codes: list[str] = Field(min_length=2, max_length=5)
    fields: list[str] = Field(default_factory=list)


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


class AfterSalesCommand(BaseModel):
    kind: Literal["create_after_sales_ticket"] = "create_after_sales_ticket"
    arguments: dict[str, Any]


class HumanHandoffCommand(BaseModel):
    kind: Literal["create_human_handoff"] = "create_human_handoff"
    arguments: dict[str, Any]


CustomerServiceCommand = Annotated[
    SearchProductsCommand
    | RecommendProductsCommand
    | CompareProductsCommand
    | KnowledgeSearchCommand
    | QueryOrderCommand
    | QueryLogisticsCommand
    | AfterSalesCommand
    | HumanHandoffCommand,
    Field(discriminator="kind"),
]


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
    tool_count: int = Field(default=0, ge=0, le=2)
    failure_reason: str | None = None
