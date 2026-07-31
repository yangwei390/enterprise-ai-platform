from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest
from backend.app.exceptions import BusinessException
from backend.app.schemas.product import ProductQuery
from backend.app.services.customer_service import (
    CustomerServiceMockService,
    default_customer_service_mock_store,
)
from backend.app.tools.base import ToolCall
from backend.app.tools.builtin.customer_service import (
    CompareProductsTool,
    CreateAfterSalesTicketTool,
    CreateHumanHandoffTool,
    ProductServiceResource,
    QueryLogisticsTool,
    QueryOrderTool,
    RecommendProductsArgs,
    RecommendProductsTool,
    SearchProductsTool,
)
from backend.app.tools.builtin.knowledge_tool import KnowledgeSearchTool
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.providers.builtin import BuiltinToolProvider
from backend.app.tools.registry import ToolDuplicateError, ToolRegistry, get_tool_registry


class FakeProductService:
    def __init__(self) -> None:
        self.list_queries: list[ProductQuery] = []
        self.recommend_queries: list[ProductQuery] = []
        self.compare_product_codes: list[list[str]] = []
        self.primary_manual_product_ids: list[list[int]] = []
        self.should_raise_business = False
        self.should_raise_unknown = False

    def list(self, query: ProductQuery):
        if self.should_raise_business:
            raise BusinessException(40020, "商品参数错误")
        if self.should_raise_unknown:
            raise RuntimeError(
                "postgresql://user:password@host/database "
                "13800138000 110101199001011234 6222021234567890123 "
                "/Volumes/private/project/file.py"
            )
        self.list_queries.append(query)
        return (
            [
                _product(
                    1,
                    "P001",
                    price=Decimal("299.00"),
                    features=["易清洗"],
                    use_cases=["宿舍"],
                )
            ],
            1,
        )

    def recommend(self, query: ProductQuery):
        self.recommend_queries.append(query)
        return (
            [
                SimpleNamespace(
                    product=_product(2, "P002", popularity_score=88),
                    score=0.876543,
                    reasons=["匹配偏好功能：低噪音"],
                )
            ],
            None,
        )

    def compare_by_product_codes(self, product_codes: list[str]):
        self.compare_product_codes.append(product_codes)
        return SimpleNamespace(
            products=[
                _product(2, "P002", price=Decimal("259.00")),
                _product(1, "P001", price=Decimal("199.00")),
            ],
            missing_product_codes=["P404"],
        )

    def primary_manual_document_ids_for_scope(
        self,
        product_ids: list[int],
        *,
        allowed_knowledge_base_ids: set[int],
    ) -> dict[int, int]:
        self.primary_manual_product_ids.append(product_ids)
        return {1: 101, 2: 202} if allowed_knowledge_base_ids else {}


def _product_provider(service: FakeProductService):
    def provider() -> ProductServiceResource:
        return ProductServiceResource(service=service)

    return provider


class CountingProductProvider:
    def __init__(self, services: list[FakeProductService]) -> None:
        self.services = services
        self.resources: list[SimpleNamespace] = []
        self.calls = 0

    def __call__(self) -> ProductServiceResource:
        service = self.services[min(self.calls, len(self.services) - 1)]
        resource = SimpleNamespace(closed=0)
        self.resources.append(resource)
        self.calls += 1
        return ProductServiceResource(
            service=service,
            close=lambda: setattr(resource, "closed", resource.closed + 1),
        )


class FakeCustomerServiceResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def __getattr__(self, name: str) -> Any:
        return self.data[name]

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return dict(self.data)


class FakeAfterSalesService:
    def __init__(self) -> None:
        self.draft_calls = 0
        self.confirm_calls = 0
        self.last_draft_request: Any = None
        self.last_confirm_request: Any = None

    def create_after_sales_draft(self, request: Any) -> FakeCustomerServiceResponse:
        self.draft_calls += 1
        self.last_draft_request = request
        return FakeCustomerServiceResponse(
            {
                "draft_id": "mock-draft-0123456789abcdef01234567",
                "operation_id": "mock-draft-0123456789abcdef01234567",
                "summary": "售后草稿",
            }
        )

    def confirm_after_sales_ticket(self, request: Any) -> FakeCustomerServiceResponse:
        self.confirm_calls += 1
        self.last_confirm_request = request
        return FakeCustomerServiceResponse(
            {
                "ticket_id": "mock-ticket-0123456789ab",
                "idempotent_replay": False,
            }
        )


def test_search_products_tool_descriptor_and_schema() -> None:
    descriptor = SearchProductsTool(_product_provider(FakeProductService())).get_descriptor()

    assert descriptor.name == "search_products"
    assert "不自动放宽条件" in descriptor.description
    assert descriptor.input_schema["additionalProperties"] is False
    assert "required_features" in descriptor.input_schema["properties"]
    assert "preferred_use_cases" in descriptor.input_schema["properties"]
    assert descriptor.input_schema["properties"]["page"]["minimum"] == 1
    assert descriptor.input_schema["properties"]["page_size"]["maximum"] == 100


def test_search_products_maps_args_to_product_query_without_mixing_preferences() -> None:
    service = FakeProductService()
    tool = SearchProductsTool(_product_provider(service))

    result = tool.run(
        {
            "product_code": "P001",
            "category": "豆浆机",
            "price_min": "200",
            "price_max": "300",
            "required_features": ["必须易清洗"],
            "excluded_features": ["高噪音"],
            "preferred_features": ["低噪音"],
            "required_use_cases": ["宿舍硬条件"],
            "preferred_use_cases": ["宿舍偏好"],
            "features": ["旧功能"],
            "use_cases": ["旧场景"],
            "sort_by": "price",
            "sort_order": "asc",
            "page": 2,
            "page_size": 5,
            "knowledge_base_id": 8,
        }
    )

    assert result.success is True
    query = service.list_queries[0]
    assert query.product_code == "P001"
    assert query.required_features == ["必须易清洗"]
    assert query.preferred_features == ["低噪音"]
    assert query.required_use_cases == ["宿舍硬条件"]
    assert query.preferred_use_cases == ["宿舍偏好"]
    assert query.features == ["旧功能"]
    assert query.use_cases == ["旧场景"]
    result_data = _result_dict(result)
    assert result_data["items"][0]["price"] == "299.00"
    assert result_data["items"][0]["primary_manual_document_id"] == 101
    assert "deleted_at" not in result_data["items"][0]


def test_recommend_products_preserves_service_score_reasons_and_order() -> None:
    service = FakeProductService()
    tool = RecommendProductsTool(_product_provider(service))

    result = tool.run({"preferred_features": ["低噪音"], "page_size": 3})
    result_data = _result_dict(result)

    assert result.success is True
    assert service.recommend_queries[0].preferred_features == ["低噪音"]
    assert result_data == {
        "items": [
            {
                "product": result_data["items"][0]["product"],
                "score": 0.876543,
                "reasons": ["匹配偏好功能：低噪音"],
            }
        ],
        "total": 1,
        "no_result_reason": None,
    }


def test_recommend_products_rejects_empty_query_conditions() -> None:
    with pytest.raises(ValueError, match="必须提供至少一个有效查询条件"):
        RecommendProductsArgs.model_validate({})


def test_recommend_products_defensively_respects_page_size() -> None:
    class MultipleProductService(FakeProductService):
        def recommend(self, query: ProductQuery):
            recommendations, reason = super().recommend(query)
            return [*recommendations, *recommendations], reason

    service = MultipleProductService()
    tool = RecommendProductsTool(_product_provider(service))

    result = tool.run({"keyword": "鼠标", "page_size": 1})
    result_data = _result_dict(result)

    assert result.success is True
    assert result_data["total"] == 1
    assert len(result_data["items"]) == 1


def test_compare_products_uses_service_batch_result_order_and_missing_codes() -> None:
    service = FakeProductService()
    tool = CompareProductsTool(_product_provider(service))

    result = tool.run(
        {
            "product_codes": ["P002", "P001", "P002", "P404"],
            "fields": ["brand", "price", "manual_evidence_summary"],
        }
    )
    result_data = _result_dict(result)

    assert result.success is True
    assert service.compare_product_codes == [["P002", "P001", "P404"]]
    assert [item["product_code"] for item in result_data["items"]] == ["P002", "P001"]
    assert result_data["missing_product_codes"] == ["P404"]
    assert result_data["items"][0]["manual_evidence_summary"] == "当前资料未提供"


def test_product_tool_business_and_unknown_errors_are_safe() -> None:
    business_service = FakeProductService()
    business_service.should_raise_business = True
    business_result = SearchProductsTool(_product_provider(business_service)).run({})

    unknown_service = FakeProductService()
    unknown_service.should_raise_unknown = True
    unknown_result = SearchProductsTool(_product_provider(unknown_service)).run({})

    assert business_result.success is False
    assert business_result.error == "商品参数错误"
    assert business_result.metadata["error_code"] == 40020
    assert unknown_result.success is False
    assert unknown_result.error == "工具执行失败"
    serialized = unknown_result.model_dump_json()
    assert "postgresql://" not in serialized
    assert "password" not in serialized
    assert "13800138000" not in serialized
    assert "110101199001011234" not in serialized
    assert "6222021234567890123" not in serialized
    assert "/Volumes/private" not in serialized


def test_product_tool_unknown_error_log_does_not_include_sensitive_exception(caplog) -> None:
    service = FakeProductService()
    service.should_raise_unknown = True

    result = SearchProductsTool(_product_provider(service)).run({})

    assert result.success is False
    log_text = caplog.text
    assert "postgresql://" not in log_text
    assert "password" not in log_text
    assert "13800138000" not in log_text
    assert "110101199001011234" not in log_text
    assert "6222021234567890123" not in log_text
    assert "/Volumes/private" not in log_text


def test_product_provider_resource_closes_on_success_business_and_unknown_error() -> None:
    success_provider = CountingProductProvider([FakeProductService()])
    SearchProductsTool(success_provider).run({})
    assert success_provider.resources[0].closed == 1

    business_service = FakeProductService()
    business_service.should_raise_business = True
    business_provider = CountingProductProvider([business_service])
    SearchProductsTool(business_provider).run({})
    assert business_provider.resources[0].closed == 1

    unknown_service = FakeProductService()
    unknown_service.should_raise_unknown = True
    unknown_provider = CountingProductProvider([unknown_service])
    SearchProductsTool(unknown_provider).run({})
    assert unknown_provider.resources[0].closed == 1


def test_product_provider_uses_independent_resource_per_execute() -> None:
    provider = CountingProductProvider([FakeProductService(), FakeProductService()])
    tool = SearchProductsTool(provider)

    first = tool.run({})
    second = tool.run({})

    assert first.success is True
    assert second.success is True
    assert provider.calls == 2
    assert len(provider.resources) == 2
    assert [resource.closed for resource in provider.resources] == [1, 1]


def test_product_tool_validation_errors_are_executor_validation_failures() -> None:
    registry = ToolRegistry()
    registry.register(SearchProductsTool(_product_provider(FakeProductService())))
    executor = ToolExecutor(registry=registry)

    for arguments in (
        {"sort_by": "deleted_at"},
        {"in_stock_only": "true"},
        {"unknown": "field"},
    ):
        result = executor.execute(ToolCall(name="search_products", arguments=arguments))

        assert result.success is False
        assert result.metadata["status"] == "validation_failed"


def test_order_and_logistics_tools_use_customer_service_and_keep_safe_error() -> None:
    store = default_customer_service_mock_store()
    service = CustomerServiceMockService(store)
    order_tool = QueryOrderTool(lambda: service)
    logistics_tool = QueryLogisticsTool(lambda: service)

    order_list = order_tool.run({})
    order_result = order_tool.run({"order_ref": "2026****0001"})
    logistics_result = logistics_tool.run({"order_ref": "2026****0001"})
    missing_order = logistics_tool.run({"order_ref": "2026****9999"})
    order_list_data = _result_dict(order_list)
    order_data = _result_dict(order_result)

    assert order_list.success is True
    assert order_list_data["mode"] == "list"
    assert order_list_data["total"] == 2
    assert order_list_data["items"][0]["order_no"] == "2026****0002"
    assert order_result.success is True
    assert order_data["phone"] == "138****5678"
    assert logistics_result.success is True
    assert missing_order.success is False
    assert missing_order.error == "订单不存在"


def test_after_sales_tool_draft_confirm_replay_and_store_sharing() -> None:
    service = CustomerServiceMockService(default_customer_service_mock_store())
    first_tool = CreateAfterSalesTicketTool(lambda: service)
    second_tool = CreateAfterSalesTicketTool(lambda: service)

    draft_result = first_tool.run(
        {
            "action": "draft",
            "order_no": "202607240001",
            "customer_phone_last4": "5678",
            "issue_type": "repair",
            "issue_description": "机器启动后有异响，需要售后检查",
        }
    )
    draft = _result_dict(draft_result)
    confirm_payload = {
        "action": "confirm",
        "order_no": "202607240001",
        "customer_phone_last4": "5678",
        "draft_id": draft["draft_id"],
        "operation_id": draft["operation_id"],
        "confirmed": True,
    }
    first_confirm = second_tool.run(confirm_payload)
    second_confirm = first_tool.run(confirm_payload)

    assert draft_result.success is True
    assert draft["status"] == "draft"
    first_confirm_data = _result_dict(first_confirm)
    second_confirm_data = _result_dict(second_confirm)
    assert first_confirm_data["ticket_id"] == second_confirm_data["ticket_id"]
    assert first_confirm_data["idempotent_replay"] is False
    assert second_confirm_data["idempotent_replay"] is True


def test_after_sales_tool_rejects_draft_with_confirm_fields_before_service_call() -> None:
    service = FakeAfterSalesService()
    registry = ToolRegistry()
    registry.register(CreateAfterSalesTicketTool(cast(Any, lambda: service)))
    executor = ToolExecutor(registry=registry)
    base_payload = {
        "action": "draft",
        "order_no": "202607240001",
        "customer_phone_last4": "5678",
        "issue_type": "repair",
        "issue_description": "机器启动后有异响，需要售后检查",
    }

    for extra in (
        {"draft_id": "mock-draft-0123456789abcdef01234567"},
        {"operation_id": "mock-draft-0123456789abcdef01234567"},
        {"confirmed": True},
        {"confirmed": False},
    ):
        result = executor.execute(
            ToolCall(name="create_after_sales_ticket", arguments={**base_payload, **extra})
        )

        assert result.success is False
        assert result.metadata["status"] == "validation_failed"

    assert service.draft_calls == 0
    assert service.confirm_calls == 0


def test_after_sales_tool_rejects_invalid_confirm_before_service_call() -> None:
    service = FakeAfterSalesService()
    registry = ToolRegistry()
    registry.register(CreateAfterSalesTicketTool(cast(Any, lambda: service)))
    executor = ToolExecutor(registry=registry)
    base_payload = {
        "action": "confirm",
        "order_no": "202607240001",
        "customer_phone_last4": "5678",
        "draft_id": "mock-draft-0123456789abcdef01234567",
        "operation_id": "mock-draft-0123456789abcdef01234567",
        "confirmed": True,
    }

    invalid_payloads = (
        {key: value for key, value in base_payload.items() if key != "draft_id"},
        {key: value for key, value in base_payload.items() if key != "operation_id"},
        {key: value for key, value in base_payload.items() if key != "confirmed"},
        {**base_payload, "confirmed": False},
        {**base_payload, "confirmed": 1},
        {**base_payload, "confirmed": "true"},
        {**base_payload, "confirmed": "yes"},
        {**base_payload, "issue_type": "repair"},
        {**base_payload, "issue_description": "客户端试图覆盖草稿内容"},
    )

    for payload in invalid_payloads:
        result = executor.execute(ToolCall(name="create_after_sales_ticket", arguments=payload))

        assert result.success is False
        assert result.metadata["status"] == "validation_failed"

    assert service.draft_calls == 0
    assert service.confirm_calls == 0


def test_after_sales_tool_valid_draft_and_confirm_call_service_once() -> None:
    service = FakeAfterSalesService()
    tool = CreateAfterSalesTicketTool(cast(Any, lambda: service))

    draft_result = tool.run(
        {
            "action": "draft",
            "order_no": "202607240001",
            "customer_phone_last4": "5678",
            "issue_type": "repair",
            "issue_description": "机器启动后有异响，需要售后检查",
        }
    )
    confirm_result = tool.run(
        {
            "action": "confirm",
            "order_no": "202607240001",
            "customer_phone_last4": "5678",
            "draft_id": "mock-draft-0123456789abcdef01234567",
            "operation_id": "mock-draft-0123456789abcdef01234567",
            "confirmed": True,
        }
    )

    assert draft_result.success is True
    assert confirm_result.success is True
    assert service.draft_calls == 1
    assert service.confirm_calls == 1
    assert service.last_draft_request.issue_type == "repair"
    assert service.last_confirm_request.confirmed is True


def test_after_sales_tool_rejects_unconfirmed_and_cross_order_operation() -> None:
    service = CustomerServiceMockService(default_customer_service_mock_store())
    tool = CreateAfterSalesTicketTool(lambda: service)
    draft = _result_dict(
        tool.run(
            {
                "action": "draft",
                "order_no": "202607240001",
                "customer_phone_last4": "5678",
                "issue_type": "repair",
                "issue_description": "机器启动后有异响，需要售后检查",
            }
        )
    )

    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)
    unconfirmed = executor.execute(
        ToolCall(
            name="create_after_sales_ticket",
            arguments={
                "action": "confirm",
                "order_no": "202607240001",
                "customer_phone_last4": "5678",
                "draft_id": draft["draft_id"],
                "operation_id": draft["operation_id"],
                "confirmed": False,
            },
        )
    )
    cross_order = tool.run(
        {
            "action": "confirm",
            "order_no": "202607240002",
            "customer_phone_last4": "1111",
            "draft_id": draft["draft_id"],
            "operation_id": draft["operation_id"],
            "confirmed": True,
        }
    )

    assert unconfirmed.success is False
    assert unconfirmed.metadata["status"] == "validation_failed"
    assert cross_order.success is False
    assert service.store.tickets == {}


def test_after_sales_tool_concurrent_confirm_uses_service_atomic_idempotency() -> None:
    service = CustomerServiceMockService(default_customer_service_mock_store())
    tool = CreateAfterSalesTicketTool(lambda: service)
    draft = _result_dict(
        tool.run(
            {
                "action": "draft",
                "order_no": "202607240001",
                "customer_phone_last4": "5678",
                "issue_type": "exchange",
                "issue_description": "并发确认测试，需要售后检查",
            }
        )
    )
    payload = {
        "action": "confirm",
        "order_no": "202607240001",
        "customer_phone_last4": "5678",
        "draft_id": draft["draft_id"],
        "operation_id": draft["operation_id"],
        "confirmed": True,
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: tool.run(payload), range(8)))

    result_dicts = [_result_dict(result) for result in results]
    assert {result_data["ticket_id"] for result_data in result_dicts} == {
        f"mock-ticket-{draft['draft_id'][-12:]}"
    }
    assert [result_data["idempotent_replay"] for result_data in result_dicts].count(False) == 1


def test_human_handoff_tool_returns_mock_without_fake_agent_or_raw_message() -> None:
    service = CustomerServiceMockService(default_customer_service_mock_store())
    tool = CreateHumanHandoffTool(lambda: service)

    result = tool.run(
        {
            "order_no": "202607240001",
            "customer_phone_last4": "5678",
            "reason": "customer_request",
            "message": "请人工联系我，手机号13800004321，地址北京市朝阳区建国路100号",
        }
    )
    result_data = _result_dict(result)

    assert result.success is True
    assert result_data["mock"] is True
    assert "agent_name" not in result_data
    assert "wait_time" not in result_data
    assert "13800004321" not in result_data["message_preview"]
    assert "建国路100号" not in result_data["message_preview"]


def test_builtin_registry_discovers_customer_service_tools_without_losing_knowledge_search(
    monkeypatch,
) -> None:
    import backend.app.tools.builtin.customer_service as customer_service_tools
    import backend.app.tools.registry as registry_module

    provider_calls = 0

    def product_provider() -> ProductServiceResource:
        nonlocal provider_calls
        provider_calls += 1
        return ProductServiceResource(service=FakeProductService())

    monkeypatch.setattr(
        customer_service_tools,
        "default_product_service_provider",
        product_provider,
    )
    monkeypatch.setattr(registry_module, "_tool_registry", None)
    registry = get_tool_registry()
    names = {descriptor.name for descriptor in registry.list_descriptors()}

    assert {
        "search_products",
        "recommend_products",
        "compare_products",
        "query_order",
        "query_logistics",
        "create_after_sales_ticket",
        "create_human_handoff",
        "knowledge_search",
    }.issubset(names)
    version_before = registry.version
    registry.refresh()
    version_after = registry.version
    registered_names = [tool.name for tool in registry.list_tools()]
    assert len(registered_names) == len(set(registered_names))
    assert version_after >= version_before
    assert provider_calls == 0
    assert isinstance(registry.get_tool("knowledge_search"), KnowledgeSearchTool)


def test_builtin_provider_refresh_is_idempotent_and_does_not_call_product_provider(
    monkeypatch,
) -> None:
    import backend.app.tools.builtin.customer_service as customer_service_tools

    provider_calls = 0

    def product_provider() -> ProductServiceResource:
        nonlocal provider_calls
        provider_calls += 1
        return ProductServiceResource(service=FakeProductService())

    monkeypatch.setattr(
        customer_service_tools,
        "default_product_service_provider",
        product_provider,
    )
    registry = ToolRegistry()
    registry.register_provider(BuiltinToolProvider())

    registry.refresh()
    names_after_first = [tool.name for tool in registry.list_tools()]
    registry.refresh()
    names_after_second = [tool.name for tool in registry.list_tools()]

    assert provider_calls == 0
    assert sorted(names_after_first) == sorted(names_after_second)
    assert len(names_after_second) == len(set(names_after_second))
    assert isinstance(registry.get_tool("knowledge_search"), KnowledgeSearchTool)


def test_tool_registry_duplicate_name_uses_existing_replace_contract() -> None:
    registry = ToolRegistry()
    original = SearchProductsTool(_product_provider(FakeProductService()))
    replacement = SearchProductsTool(_product_provider(FakeProductService()))

    registry.register(original)
    with pytest.raises(ToolDuplicateError):
        registry.register(replacement)

    registry.register(replacement, replace=True)
    assert registry.get_tool("search_products") is replacement


def _product(
    id: int,
    product_code: str,
    *,
    price: Decimal = Decimal("199.00"),
    popularity_score: int = 50,
    features: list[str] | None = None,
    use_cases: list[str] | None = None,
):
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=id,
        product_code=product_code,
        brand="九阳",
        name=f"商品 {product_code}",
        model=f"M{id}",
        category="豆浆机",
        description="模拟商品",
        price=price,
        currency="CNY",
        stock_quantity=5,
        sale_status="on_sale",
        features=features or [],
        use_cases=use_cases or [],
        specifications={"capacity": "1L"},
        tags=["模拟"],
        popularity_score=popularity_score,
        official_product_url=None,
        source_checked_at=now,
        is_active=True,
    )


def _result_dict(result) -> dict[str, Any]:
    assert isinstance(result.result, dict)
    return cast(dict[str, Any], result.result)
