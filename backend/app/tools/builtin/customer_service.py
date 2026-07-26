from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.schemas.customer_service import (
    AfterSalesConfirmRequest,
    AfterSalesDraftRequest,
    AfterSalesIssueType,
    CustomerOrderQuery,
    HandoffReason,
    HumanHandoffRequest,
)
from backend.app.schemas.product import ProductQuery
from backend.app.services.customer_service import CustomerServiceMockService
from backend.app.tools.base import BaseTool, ToolResult
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

SortBy = Literal["popularity", "price", "stock_quantity", "created_at", "updated_at"]
SortOrder = Literal["asc", "desc"]
SaleStatus = Literal["on_sale", "off_sale", "pre_sale", "discontinued"]
AfterSalesAction = Literal["draft", "confirm"]
CompareField = Literal[
    "brand",
    "model",
    "category",
    "price",
    "stock_quantity",
    "sale_status",
    "features",
    "use_cases",
    "specifications",
    "tags",
    "manual_evidence_summary",
]

COMPARE_FIELD_DEFAULTS: list[CompareField] = [
    "brand",
    "model",
    "category",
    "price",
    "stock_quantity",
    "sale_status",
    "features",
    "use_cases",
    "specifications",
    "tags",
    "manual_evidence_summary",
]


class ProductToolQueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    keyword: str | None = Field(default=None, max_length=128)
    brand: str | None = Field(default=None, max_length=128)
    category: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=128)
    price_min: Decimal | None = Field(default=None, ge=0)
    price_max: Decimal | None = Field(default=None, ge=0)
    required_features: list[str] = Field(default_factory=list, max_length=20)
    excluded_features: list[str] = Field(default_factory=list, max_length=20)
    preferred_features: list[str] = Field(default_factory=list, max_length=20)
    required_use_cases: list[str] = Field(default_factory=list, max_length=20)
    preferred_use_cases: list[str] = Field(default_factory=list, max_length=20)
    features: list[str] = Field(default_factory=list, max_length=20)
    use_cases: list[str] = Field(default_factory=list, max_length=20)
    in_stock_only: StrictBool = True
    sale_status: SaleStatus | None = "on_sale"
    sort_by: SortBy = "popularity"
    sort_order: SortOrder = "desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    knowledge_base_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "平台注入的当前知识库范围，用于限定主说明书 document_id；"
            "缺失时不返回说明书 ID。"
        ),
    )

    @field_validator(
        "required_features",
        "excluded_features",
        "preferred_features",
        "required_use_cases",
        "preferred_use_cases",
        "features",
        "use_cases",
    )
    @classmethod
    def validate_text_list(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("列表项不能为空")
            if len(cleaned) > 64:
                raise ValueError("列表项不能超过 64 个字符")
            result.append(cleaned)
        return result

    @model_validator(mode="after")
    def validate_price_range(self) -> ProductToolQueryArgs:
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min 不能大于 price_max")
        return self

    def to_product_query(self) -> ProductQuery:
        return ProductQuery(**self.model_dump(exclude={"knowledge_base_id"}))

    def manual_knowledge_base_scope(self) -> set[int]:
        return {self.knowledge_base_id} if self.knowledge_base_id is not None else set()


class SearchProductsArgs(ProductToolQueryArgs):
    pass


class RecommendProductsArgs(ProductToolQueryArgs):
    page_size: int = Field(default=3, ge=1, le=100)


class CompareProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_codes: list[str] = Field(min_length=2, max_length=5)
    fields: list[CompareField] = Field(default_factory=lambda: list(COMPARE_FIELD_DEFAULTS))
    knowledge_base_id: int | None = Field(default=None, ge=1)

    def manual_knowledge_base_scope(self) -> set[int]:
        return {self.knowledge_base_id} if self.knowledge_base_id is not None else set()

    @field_validator("product_codes")
    @classmethod
    def validate_product_codes(cls, values: list[str]) -> list[str]:
        cleaned_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("product_codes 不能包含空值")
            if len(cleaned) > 64:
                raise ValueError("product_code 不能超过 64 个字符")
            if cleaned not in seen:
                cleaned_values.append(cleaned)
                seen.add(cleaned)
        if len(cleaned_values) < 2:
            raise ValueError("compare_products 至少需要 2 个不同商品")
        return cleaned_values


class AfterSalesTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: AfterSalesAction = Field(
        default="draft",
        description="两阶段操作：draft 创建售后草稿；confirm 确认已有草稿并创建 Mock 工单。",
    )
    order_no: str = Field(min_length=4, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")
    customer_phone_last4: str = Field(pattern=r"^\d{4}$")
    issue_type: AfterSalesIssueType | None = Field(
        default=None,
        description="仅 action=draft 使用；action=confirm 不得传入。",
    )
    issue_description: str | None = Field(
        default=None,
        min_length=5,
        max_length=1000,
        description="仅 action=draft 使用；action=confirm 不得传入。",
    )
    draft_id: str | None = Field(
        default=None,
        min_length=35,
        max_length=35,
        pattern=r"^mock-draft-[0-9a-f]{24}$",
        description="仅 action=confirm 使用；action=draft 不得传入。",
    )
    operation_id: str | None = Field(
        default=None,
        min_length=35,
        max_length=35,
        pattern=r"^mock-draft-[0-9a-f]{24}$",
        description="仅 action=confirm 使用；action=draft 不得传入。",
    )
    confirmed: StrictBool | None = Field(
        default=None,
        description="仅 action=confirm 使用，且必须为严格布尔 true。",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> AfterSalesTicketArgs:
        if self.action == "draft":
            forbidden_fields = {"draft_id", "operation_id", "confirmed"}
            if forbidden_fields & self.model_fields_set:
                raise ValueError("创建售后草稿不能传入 draft_id、operation_id 或 confirmed")
            if self.issue_type is None or self.issue_description is None:
                raise ValueError("创建售后草稿需要 issue_type 和 issue_description")
            return self
        forbidden_fields = {"issue_type", "issue_description"}
        if forbidden_fields & self.model_fields_set:
            raise ValueError("确认售后工单不能传入 issue_type 或 issue_description")
        if self.draft_id is None or self.operation_id is None or self.confirmed is None:
            raise ValueError("确认售后工单需要 draft_id、operation_id 和 confirmed")
        if self.confirmed is not True:
            raise ValueError("确认售后工单时 confirmed 必须为 true")
        return self

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class HumanHandoffToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str = Field(min_length=4, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")
    customer_phone_last4: str = Field(pattern=r"^\d{4}$")
    reason: HandoffReason = "customer_request"
    message: str = Field(min_length=2, max_length=1000)


@dataclass(slots=True)
class ProductServiceResource:
    service: Any
    close: Callable[[], None] = lambda: None


ProductServiceProvider = Callable[[], ProductServiceResource]
CustomerServiceProvider = Callable[[], CustomerServiceMockService]


def default_product_service_provider() -> ProductServiceResource:
    from backend.app.db.session import SessionLocal
    from backend.app.repositories.product import ProductRepository
    from backend.app.services.product import ProductService

    db = SessionLocal()
    return ProductServiceResource(
        service=ProductService(ProductRepository(db)),
        close=db.close,
    )


def default_customer_service_provider() -> CustomerServiceMockService:
    return CustomerServiceMockService()


class _ProductTool(BaseTool):
    source = "builtin"
    permission = "public"

    def __init__(self, product_service_provider: ProductServiceProvider | None = None) -> None:
        self.product_service_provider = product_service_provider or default_product_service_provider

    def _with_product_service(self, callback):
        resource = self.product_service_provider()
        try:
            return callback(resource.service)
        finally:
            resource.close()


class SearchProductsTool(_ProductTool):
    name = "search_products"
    description = (
        "查询模拟商品目录，支持硬条件、软偏好、分页和排序；"
        "只返回商品数据，不自动放宽条件。"
    )
    args_schema = SearchProductsArgs

    def run(self, arguments: dict) -> ToolResult:
        args = SearchProductsArgs.model_validate(arguments)

        def callback(service):
            products, total = service.list(args.to_product_query())
            manual_document_ids = service.primary_manual_document_ids_for_scope(
                [product.id for product in products],
                allowed_knowledge_base_ids=args.manual_knowledge_base_scope(),
            )
            return ToolResult(
                name=self.name,
                success=True,
                result={
                    "items": [
                        _product_to_tool_item(
                            product,
                            primary_manual_document_id=manual_document_ids.get(product.id),
                        )
                        for product in products
                    ],
                    "total": total,
                    "page": args.page,
                    "page_size": args.page_size,
                },
                metadata={"failed": False},
            )

        return _safe_tool_call(self.name, lambda: self._with_product_service(callback))


class RecommendProductsTool(_ProductTool):
    name = "recommend_products"
    description = (
        "按 ProductService 的确定性硬过滤和评分规则推荐商品；"
        "保留 score、reasons 和无结果原因。"
    )
    args_schema = RecommendProductsArgs

    def run(self, arguments: dict) -> ToolResult:
        args = RecommendProductsArgs.model_validate(arguments)

        def callback(service):
            recommendations, no_result_reason = service.recommend(args.to_product_query())
            manual_document_ids = service.primary_manual_document_ids_for_scope(
                [item.product.id for item in recommendations],
                allowed_knowledge_base_ids=args.manual_knowledge_base_scope(),
            )
            items = [
                {
                    "product": _product_to_tool_item(
                        item.product,
                        primary_manual_document_id=manual_document_ids.get(item.product.id),
                    ),
                    "score": item.score,
                    "reasons": list(item.reasons),
                }
                for item in recommendations
            ]
            return ToolResult(
                name=self.name,
                success=True,
                result={
                    "items": items,
                    "total": len(items),
                    "no_result_reason": no_result_reason,
                },
                metadata={"failed": False},
            )

        return _safe_tool_call(self.name, lambda: self._with_product_service(callback))


class CompareProductsTool(_ProductTool):
    name = "compare_products"
    description = (
        "对用户明确指定的商品编码做字段级对比；"
        "缺失字段标记为当前资料未提供，不臆造商品能力。"
    )
    args_schema = CompareProductsArgs

    def run(self, arguments: dict) -> ToolResult:
        args = CompareProductsArgs.model_validate(arguments)

        def callback(service):
            comparison = service.compare_by_product_codes(args.product_codes)
            manual_document_ids = service.primary_manual_document_ids_for_scope(
                [product.id for product in comparison.products],
                allowed_knowledge_base_ids=args.manual_knowledge_base_scope(),
            )
            return ToolResult(
                name=self.name,
                success=True,
                result={
                    "items": [
                        _product_to_comparison_item(
                            product,
                            args.fields,
                            primary_manual_document_id=manual_document_ids.get(product.id),
                        )
                        for product in comparison.products
                    ],
                    "missing_product_codes": comparison.missing_product_codes,
                },
                metadata={"failed": False},
            )

        return _safe_tool_call(self.name, lambda: self._with_product_service(callback))


class _CustomerServiceTool(BaseTool):
    source = "builtin"
    permission = "public"

    def __init__(self, customer_service_provider: CustomerServiceProvider | None = None) -> None:
        self.customer_service_provider = (
            customer_service_provider or default_customer_service_provider
        )

    @property
    def customer_service(self) -> CustomerServiceMockService:
        return self.customer_service_provider()


class QueryOrderTool(_CustomerServiceTool):
    name = "query_order"
    description = "查询 Mock 订单状态；必须提供订单号和手机号后四位，返回内容已由 Service 脱敏。"
    args_schema = CustomerOrderQuery

    def run(self, arguments: dict) -> ToolResult:
        args = CustomerOrderQuery.model_validate(arguments)
        return _safe_tool_call(
            self.name,
            lambda: ToolResult(
                name=self.name,
                success=True,
                result=self.customer_service.query_order(args).model_dump(mode="json"),
                metadata={"mock": True, "failed": False},
            ),
        )


class QueryLogisticsTool(_CustomerServiceTool):
    name = "query_logistics"
    description = "查询 Mock 物流状态；必须先通过订单归属校验，不提供实时物流承诺。"
    args_schema = CustomerOrderQuery

    def run(self, arguments: dict) -> ToolResult:
        args = CustomerOrderQuery.model_validate(arguments)
        return _safe_tool_call(
            self.name,
            lambda: ToolResult(
                name=self.name,
                success=True,
                result=self.customer_service.query_logistics(args).model_dump(mode="json"),
                metadata={"mock": True, "failed": False},
            ),
        )


class CreateAfterSalesTicketTool(_CustomerServiceTool):
    name = "create_after_sales_ticket"
    description = (
        "创建模拟售后工单的两阶段工具。action=draft 只返回草稿摘要；"
        "action=confirm 必须携带 draft_id、operation_id 且 confirmed=true 后才创建 Mock 工单。"
    )
    args_schema = AfterSalesTicketArgs

    def run(self, arguments: dict) -> ToolResult:
        args = AfterSalesTicketArgs.model_validate(arguments)
        return _safe_tool_call(self.name, lambda: self._run_after_sales(args))

    def _run_after_sales(self, args: AfterSalesTicketArgs) -> ToolResult:
        if args.action == "draft":
            draft = self.customer_service.create_after_sales_draft(
                AfterSalesDraftRequest(
                    order_no=args.order_no,
                    customer_phone_last4=args.customer_phone_last4,
                    issue_type=args.issue_type or "other",
                    issue_description=args.issue_description or "",
                )
            )
            return ToolResult(
                name=self.name,
                success=True,
                result={"status": "draft", **draft.model_dump(mode="json")},
                metadata={"mock": True, "requires_confirmation": True, "failed": False},
            )

        ticket = self.customer_service.confirm_after_sales_ticket(
            AfterSalesConfirmRequest(
                order_no=args.order_no,
                customer_phone_last4=args.customer_phone_last4,
                draft_id=args.draft_id or "",
                operation_id=args.operation_id or "",
                confirmed=args.confirmed if args.confirmed is not None else False,
            )
        )
        return ToolResult(
            name=self.name,
            success=True,
            result={"status": "created", **ticket.model_dump(mode="json")},
            metadata={
                "mock": True,
                "idempotent_replay": ticket.idempotent_replay,
                "failed": False,
            },
        )


class CreateHumanHandoffTool(_CustomerServiceTool):
    name = "create_human_handoff"
    description = "创建模拟转人工记录；不伪造客服姓名、等待时间、已接通或处理结果。"
    args_schema = HumanHandoffToolArgs

    def run(self, arguments: dict) -> ToolResult:
        args = HumanHandoffToolArgs.model_validate(arguments)
        return _safe_tool_call(
            self.name,
            lambda: ToolResult(
                name=self.name,
                success=True,
                result=self.customer_service.create_human_handoff(
                    HumanHandoffRequest(**args.model_dump())
                ).model_dump(mode="json"),
                metadata={"mock": True, "failed": False},
            ),
        )


def _safe_tool_call(tool_name: str, callback: Callable[[], ToolResult]) -> ToolResult:
    try:
        return callback()
    except BusinessException as exc:
        return ToolResult(
            name=tool_name,
            success=False,
            error=exc.message,
            metadata={
                "failed": True,
                "error_code": exc.code,
                "error_type": "business_error",
            },
        )
    except Exception:
        logger.warning("Customer service tool failed | tool=%s", tool_name)
        return ToolResult(
            name=tool_name,
            success=False,
            error="工具执行失败",
            metadata={"failed": True, "error_type": "tool_runtime_error"},
        )


def _product_to_tool_item(
    product: Any,
    *,
    primary_manual_document_id: int | None = None,
) -> dict[str, Any]:
    return _jsonable(
        {
            "id": product.id,
            "product_code": product.product_code,
            "brand": product.brand,
            "name": product.name,
            "model": product.model,
            "category": product.category,
            "description": getattr(product, "description", None),
            "price": product.price,
            "currency": product.currency,
            "stock_quantity": product.stock_quantity,
            "sale_status": product.sale_status,
            "features": getattr(product, "features", []),
            "use_cases": getattr(product, "use_cases", []),
            "specifications": getattr(product, "specifications", {}),
            "tags": getattr(product, "tags", []),
            "popularity_score": product.popularity_score,
            "official_product_url": getattr(product, "official_product_url", None),
            "source_checked_at": getattr(product, "source_checked_at", None),
            "is_active": product.is_active,
            "primary_manual_document_id": primary_manual_document_id,
        }
    )


def _product_to_comparison_item(
    product: Any,
    fields: list[CompareField],
    *,
    primary_manual_document_id: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "product_code": product.product_code,
        "name": product.name,
        "primary_manual_document_id": primary_manual_document_id,
    }
    for field in fields:
        if field == "manual_evidence_summary":
            item[field] = "当前资料未提供"
            continue
        value = getattr(product, field, None)
        item[field] = _missing_if_empty(value)
    return _jsonable(item)


def _missing_if_empty(value: Any) -> Any:
    if value is None or value == "" or value == [] or value == {}:
        return "当前资料未提供"
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
