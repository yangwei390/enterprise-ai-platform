from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from backend.app.api.product import get_product_service
from backend.app.api.product import router as product_router
from backend.app.schemas.product import ProductCreate, ProductQuery, ProductUpdate
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeProductService:
    def __init__(self) -> None:
        self.last_query: ProductQuery | None = None
        self.created: ProductCreate | None = None
        self.updated: ProductUpdate | None = None
        self.linked_document = None
        self.allowed_knowledge_base_ids: set[int] | None = None
        self.called = False

    def list(self, query: ProductQuery):
        self.called = True
        self.last_query = query
        return [_product(1, "P001")], 1

    def recommend(self, query: ProductQuery):
        self.called = True
        self.last_query = query
        return [
            SimpleNamespace(
                product=_product(1, "P001"),
                score=0.8,
                reasons=["满足全部硬条件"],
            )
        ], None

    def create(self, data: ProductCreate):
        self.created = data
        return _product(2, data.product_code)

    def get(self, id: int):
        return _product(id, "P001")

    def update(self, id: int, data: ProductUpdate):
        self.updated = data
        product = _product(id, "P001")
        if data.name is not None:
            product.name = data.name
        return product

    def deactivate(self, id: int) -> None:
        return None

    def list_document_links(self, product_id: int):
        now = datetime.now(UTC)
        return [
            SimpleNamespace(
                id=1,
                product_id=product_id,
                document_id=10,
                document_type="manual",
                is_primary=True,
                manual_version="v1",
                source_url=None,
                downloaded_at=None,
                created_at=now,
                updated_at=now,
            )
        ]

    def link_document(self, data, *, allowed_knowledge_base_ids: set[int]):
        now = datetime.now(UTC)
        self.linked_document = data
        self.allowed_knowledge_base_ids = allowed_knowledge_base_ids
        return SimpleNamespace(
            id=2,
            product_id=2,
            document_id=data.document_id,
            document_type=data.document_type,
            is_primary=data.is_primary,
            manual_version=data.manual_version,
            source_url=None,
            downloaded_at=None,
            created_at=now,
            updated_at=now,
        )


def test_product_list_api_uses_service_and_converts_legacy_fields() -> None:
    service = FakeProductService()
    client = _client(service)

    response = client.get(
        "/products",
        params=[
            ("features", "易清洗"),
            ("use_cases", "宿舍"),
            ("preferred_features", "低噪音"),
        ],
    )

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["data"]["total"] == 1
    assert service.last_query is not None
    assert service.last_query.features == ["易清洗"]
    assert service.last_query.use_cases == ["宿舍"]
    assert service.last_query.preferred_features == ["低噪音"]


def test_product_recommendation_api_returns_scores_and_reasons() -> None:
    service = FakeProductService()
    client = _client(service)

    response = client.get(
        "/products/recommendations",
        params=[("preferred_use_cases", "宿舍")],
    )

    body = response.json()
    assert response.status_code == 200
    assert body["data"]["items"][0]["score"] == 0.8
    assert body["data"]["items"][0]["reasons"] == ["满足全部硬条件"]
    assert service.last_query is not None
    assert service.last_query.preferred_use_cases == ["宿舍"]


def test_product_create_update_delete_and_document_routes(monkeypatch) -> None:
    service = FakeProductService()
    client = _client(service)
    monkeypatch.setattr(
        "backend.app.api.product.settings.CUSTOMER_SERVICE_ALLOWED_KNOWLEDGE_BASE_IDS",
        "8",
    )

    create_response = client.post(
        "/products",
        json={
            "product_code": "P002",
            "brand": "罗技",
            "name": "模拟鼠标",
            "model": "MOUSE-2",
            "category": "鼠标和指针设备",
            "price": "299.00",
        },
    )
    update_response = client.put("/products/2", json={"name": "更新后商品"})
    delete_response = client.delete("/products/2")
    links_response = client.get("/products/2/documents")
    link_response = client.post(
        "/products/2/documents",
        json={
            "document_id": 10,
            "document_type": "manual",
            "is_primary": True,
            "manual_version": "v1",
        },
    )

    assert create_response.status_code == 200
    assert create_response.json()["data"]["product_code"] == "P002"
    assert service.created is not None
    assert update_response.json()["data"]["name"] == "更新后商品"
    assert service.updated is not None
    assert delete_response.json()["data"]["deleted"] is True
    assert links_response.json()["data"]["items"][0]["document_id"] == 10
    assert link_response.json()["data"]["document_id"] == 10
    assert link_response.json()["data"]["is_primary"] is True
    assert service.linked_document is not None
    assert service.linked_document.product_code == "P001"
    assert service.allowed_knowledge_base_ids == {8}


def test_product_api_rejects_nonstandard_category_before_service_call() -> None:
    service = FakeProductService()
    response = _client(service).post(
        "/products",
        json={
            "product_code": "P002",
            "brand": "罗技",
            "name": "模拟鼠标",
            "model": "MOUSE-2",
            "category": "办公用品",
            "price": "299.00",
        },
    )

    assert response.status_code == 422
    assert service.created is None


def test_product_api_rejects_invalid_sort_before_service_call() -> None:
    _assert_invalid_query_is_422("/products", {"sort_by": "deleted_at"})


def test_product_api_query_validation_errors_return_422_before_service_call() -> None:
    invalid_cases = [
        {"price_min": "300", "price_max": "200"},
        {"page": "0"},
        {"page_size": "0"},
        {"page_size": "101"},
        {"sale_status": "invalid"},
        {"sort_by": "deleted_at"},
        {"sort_order": "sideways"},
    ]

    for params in invalid_cases:
        _assert_invalid_query_is_422("/products", params)
        _assert_invalid_query_is_422("/products/recommendations", params)


def test_product_api_normal_recommendation_request_still_returns_200() -> None:
    service = FakeProductService()
    client = _client(service)

    response = client.get(
        "/products/recommendations",
        params={"sort_by": "price", "sort_order": "asc"},
    )

    assert response.status_code == 200
    assert service.called is True


def _assert_invalid_query_is_422(path: str, params: dict[str, str]) -> None:
    service = FakeProductService()
    client = _client(service)

    response = client.get(path, params=params)

    assert response.status_code == 422
    assert service.called is False


def _client(service: FakeProductService) -> TestClient:
    app = FastAPI()
    app.include_router(product_router)
    app.dependency_overrides[get_product_service] = lambda: service
    return TestClient(app)


def _product(id: int, product_code: str):
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=id,
        product_code=product_code,
        brand="九阳",
        name=f"商品 {product_code}",
        model=f"M{id}",
        category="豆浆机",
        description="模拟商品",
        price=Decimal("199.00"),
        currency="CNY",
        stock_quantity=5,
        sale_status="on_sale",
        features=["易清洗"],
        use_cases=["宿舍"],
        specifications={},
        tags=[],
        popularity_score=70,
        official_product_url=None,
        source_checked_at=None,
        is_active=True,
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )
