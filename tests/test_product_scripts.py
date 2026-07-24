from __future__ import annotations

import json
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest
from backend.app.exceptions import BusinessException
from backend.app.schemas.product import ProductDocumentLinkCreate
from backend.app.services.product import PRODUCT_ERROR_NOT_FOUND
from scripts import seed_customer_service_products
from scripts.link_product_manual import run_link_product_manual
from scripts.seed_customer_service_products import load_fixture, run_seed


class FakeSeedService:
    def __init__(self) -> None:
        self.products: dict[str, SimpleNamespace] = {}
        self.created = 0
        self.updated = 0
        self.rollbacks = 0
        self.fail_create_codes: set[str] = set()
        self.fail_update_codes: set[str] = set()

    def get_by_product_code(self, product_code: str, *, include_deleted: bool = False):
        product = self.products.get(product_code)
        if product is None:
            raise BusinessException(PRODUCT_ERROR_NOT_FOUND, "商品不存在")
        return product

    def create(self, data):
        if data.product_code in self.fail_create_codes:
            raise RuntimeError("create failed")
        self.created += 1
        product = SimpleNamespace(id=len(self.products) + 1, **data.model_dump())
        self.products[data.product_code] = product
        return product

    def update(self, id: int, data):
        product = next(item for item in self.products.values() if item.id == id)
        if product.product_code in self.fail_update_codes:
            raise RuntimeError("update failed")
        self.updated += 1
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(product, key, value)
        return product

    def rollback(self) -> None:
        self.rollbacks += 1


class FakeManualLinkService:
    def __init__(self) -> None:
        self.calls = 0

    def link_document(self, data, *, allowed_knowledge_base_ids):
        self.calls += 1
        return SimpleNamespace(
            id=31,
            product_code=data.product_code,
            document_id=data.document_id,
        )


def test_seed_creates_product_then_skips_second_run(tmp_path) -> None:
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "模拟豆浆机",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
                "features": ["易清洗"],
                "use_cases": ["宿舍"],
            }
        ],
    )
    service = FakeSeedService()

    first = run_seed(fixture_path, service, dry_run=False)
    second = run_seed(fixture_path, service, dry_run=False)

    assert first.created == 1
    assert second.skipped == 1
    assert service.created == 1


def test_seed_updates_stable_fields_without_deleting_manual_data(tmp_path) -> None:
    service = FakeSeedService()
    service.products["P001"] = SimpleNamespace(
        id=1,
        product_code="P001",
        brand="九阳",
        name="旧名称",
        model="M1",
        category="豆浆机",
        description=None,
        price=Decimal("199.00"),
        currency="CNY",
        stock_quantity=0,
        sale_status="on_sale",
        features=[],
        use_cases=[],
        specifications={},
        tags=[],
        popularity_score=0,
        official_product_url=None,
        source_checked_at=None,
        is_active=True,
    )
    service.products["MANUAL"] = SimpleNamespace(id=2, product_code="MANUAL")
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "新名称",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
            }
        ],
    )

    stats = run_seed(fixture_path, service, dry_run=False)

    assert stats.updated == 1
    assert service.products["P001"].name == "新名称"
    assert "MANUAL" in service.products


def test_seed_dry_run_does_not_write(tmp_path) -> None:
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "模拟豆浆机",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
            }
        ],
    )
    service = FakeSeedService()

    stats = run_seed(fixture_path, service, dry_run=True)

    assert stats.created == 1
    assert service.created == 0
    assert service.products == {}


def test_seed_apply_counts_item_failures_and_continues(tmp_path) -> None:
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "模拟豆浆机 1",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
            },
            {
                "product_code": "P002",
                "brand": "九阳",
                "name": "模拟豆浆机 2",
                "model": "M2",
                "category": "豆浆机",
                "price": "299.00",
            },
            {
                "product_code": "P003",
                "brand": "九阳",
                "name": "模拟豆浆机 3",
                "model": "M3",
                "category": "豆浆机",
                "price": "399.00",
            },
        ],
    )
    service = FakeSeedService()
    service.fail_create_codes.add("P002")

    stats = run_seed(fixture_path, service, dry_run=False)

    assert stats.created == 2
    assert stats.failed == 1
    assert service.created == 2
    assert service.rollbacks == 1
    assert "P001" in service.products
    assert "P003" in service.products


def test_seed_dry_run_does_not_call_write_methods_for_items_that_would_fail(tmp_path) -> None:
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "模拟豆浆机",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
            }
        ],
    )
    service = FakeSeedService()
    service.fail_create_codes.add("P001")

    stats = run_seed(fixture_path, service, dry_run=True)

    assert stats.created == 1
    assert stats.failed == 0
    assert service.created == 0
    assert service.rollbacks == 0


def test_seed_cli_returns_non_zero_when_any_item_failed(monkeypatch, tmp_path) -> None:
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "模拟豆浆机",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
            }
        ],
    )
    service = FakeSeedService()
    service.fail_create_codes.add("P001")
    monkeypatch.setattr(seed_customer_service_products, "build_service", lambda: service)
    monkeypatch.setattr(
        sys,
        "argv",
        ["seed_customer_service_products.py", str(fixture_path), "--apply"],
    )

    exit_code = seed_customer_service_products.main()

    assert exit_code == 1
    assert service.rollbacks == 1


def test_seed_fixture_rejects_document_id(tmp_path) -> None:
    fixture_path = _fixture(
        tmp_path,
        [
            {
                "product_code": "P001",
                "brand": "九阳",
                "name": "模拟豆浆机",
                "model": "M1",
                "category": "豆浆机",
                "price": "199.00",
                "document_id": 99,
            }
        ],
    )

    with pytest.raises(ValueError, match="document_id"):
        load_fixture(fixture_path)


def test_manual_link_dry_run_does_not_call_service() -> None:
    service = FakeManualLinkService()
    payload = ProductDocumentLinkCreate(
        product_code="P001",
        document_id=11,
        document_type="manual",
        is_primary=True,
    )

    result = run_link_product_manual(
        payload,
        service,
        allowed_knowledge_base_ids={7},
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.link_id is None
    assert service.calls == 0


def test_manual_link_apply_calls_service_with_allowed_knowledge_bases() -> None:
    service = FakeManualLinkService()
    payload = ProductDocumentLinkCreate(
        product_code="P001",
        document_id=11,
        document_type="manual",
        is_primary=True,
    )

    result = run_link_product_manual(
        payload,
        service,
        allowed_knowledge_base_ids={7},
        dry_run=False,
    )

    assert result.link_id == 31
    assert service.calls == 1


def _fixture(tmp_path, products):
    path = tmp_path / "products.json"
    path.write_text(json.dumps({"products": products}), encoding="utf-8")
    return path
