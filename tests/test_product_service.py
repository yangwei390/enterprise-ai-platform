from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest
from backend.app.exceptions import BusinessException
from backend.app.repositories.product import ProductListFilters, ProductRepository
from backend.app.schemas.product import ProductDocumentLinkCreate, ProductQuery
from backend.app.services.product import (
    PRODUCT_ERROR_DOCUMENT_LINK,
    ProductService,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError


class FakeProductRepository:
    def __init__(self) -> None:
        self.filters: ProductListFilters | None = None
        self.committed = False
        self.rolled_back = False
        self.links = []
        self.primary_unset_for: int | None = None
        self.calls: list[str] = []
        self.raise_on_create: Exception | None = None
        self.products = {
            "P001": _product(1, "P001", features=["易清洗"], use_cases=["宿舍"]),
        }
        self.documents = {
            11: SimpleNamespace(
                id=11,
                knowledge_base_id=7,
                deleted_at=None,
                parse_status="success",
            ),
            12: SimpleNamespace(
                id=12,
                knowledge_base_id=7,
                deleted_at=None,
                parse_status="pending",
            ),
        }

    def get(self, id: int, *, include_deleted: bool = False):
        for product in self.products.values():
            if product.id == id and (include_deleted or product.deleted_at is None):
                return product
        return None

    def get_for_update(self, id: int):
        self.calls.append("lock_product")
        return self.get(id)

    def get_by_product_code(self, product_code: str, *, include_deleted: bool = False):
        product = self.products.get(product_code)
        if product and (include_deleted or product.deleted_at is None):
            return product
        return None

    def create_product(self, data, *, commit: bool = True):
        product = _product(len(self.products) + 1, data["product_code"])
        self.products[product.product_code] = product
        return product

    def update_product(self, product, data, *, commit: bool = True):
        for key, value in data.items():
            setattr(product, key, value)
        return product

    def list_products(self, filters, *, offset, limit, sort_column, sort_order):
        self.filters = filters
        products = [
            _product(
                1,
                "P001",
                price=Decimal("199.00"),
                popularity_score=70,
                features=["易清洗", "低噪音"],
                use_cases=["宿舍"],
            ),
            _product(
                2,
                "P002",
                price=Decimal("259.00"),
                popularity_score=70,
                features=["低噪音"],
                use_cases=["家庭"],
            ),
        ]
        return products[offset : offset + limit], len(products)

    def list_document_links(self, product_id: int):
        return self.links

    def get_document_link(self, product_id: int, document_id: int):
        for link in self.links:
            if link.product_id == product_id and link.document_id == document_id:
                return link
        return None

    def create_document_link(self, data, *, flush: bool = True):
        self.calls.append(
            "create_non_primary" if data.get("is_primary") is False else "create_primary"
        )
        if self.raise_on_create is not None:
            raise self.raise_on_create
        link = SimpleNamespace(id=len(self.links) + 1, **data)
        self.links.append(link)
        return link

    def get_primary_manual_link(self, product_id: int):
        for link in self.links:
            if link.product_id == product_id and link.document_type == "manual" and link.is_primary:
                return link
        return None

    def get_primary_manual_link_for_update(self, product_id: int):
        self.calls.append("lock_old_primary")
        return self.get_primary_manual_link(product_id)

    def unset_primary_manual_link(self, link):
        self.calls.append("unset_old")
        self.primary_unset_for = link.product_id
        link.is_primary = False

    def set_primary_manual_link(self, link):
        self.calls.append("set_new_primary")
        link.is_primary = True

    def get_active_document(self, document_id: int):
        return self.documents.get(document_id)

    def commit(self) -> None:
        self.calls.append("commit")
        self.committed = True

    def flush(self) -> None:
        self.calls.append("flush")

    def rollback(self) -> None:
        self.calls.append("rollback")
        self.rolled_back = True

    def begin_nested(self):
        repository = self

        class NestedTransaction:
            def __enter__(self):
                repository.calls.append("begin_nested")
                return self

            def __exit__(self, exc_type, exc, tb):
                repository.calls.append(
                    "savepoint_rollback" if exc_type is not None else "savepoint_release"
                )
                return False

        return NestedTransaction()

    def refresh(self, obj) -> None:
        return None


def test_product_service_converts_legacy_features_and_use_cases() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)

    service.list(
        ProductQuery(
            features=["易清洗"],
            use_cases=["宿舍"],
            preferred_use_cases=["小户型"],
            required_use_cases=["办公室"],
        )
    )

    assert repository.filters is not None
    assert repository.filters.required_features == ["易清洗"]
    assert repository.filters.required_use_cases == ["办公室"]
    assert "宿舍" not in repository.filters.required_use_cases

    normalized = service.normalize_query(ProductQuery(features=["易清洗"], use_cases=["宿舍"]))
    assert normalized.required_features == ["易清洗"]
    assert normalized.preferred_use_cases == ["宿舍"]


def test_product_service_rejects_non_allowlisted_sort_field() -> None:
    with pytest.raises(ValueError, match="sort_by 不在白名单中"):
        ProductQuery(sort_by="deleted_at")


def test_product_service_recommendation_normalizes_score_and_uses_stable_order() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)

    recommendations, no_result_reason = service.recommend(
        ProductQuery(
            category="豆浆机",
            preferred_features=["低噪音"],
            preferred_use_cases=["宿舍"],
        )
    )

    assert no_result_reason is None
    assert len(recommendations) == 2
    assert all(0 <= item.score <= 1 for item in recommendations)
    assert recommendations[0].product.product_code == "P001"


def test_product_repository_does_not_own_recommendation_or_nlp_rules() -> None:
    assert not hasattr(ProductRepository, "recommend")
    assert not hasattr(ProductRepository, "parse_user_query")


def test_link_document_rejects_unparsed_document() -> None:
    service = _product_service(FakeProductRepository())

    with pytest.raises(BusinessException) as exc_info:
        service.link_document(
            ProductDocumentLinkCreate(
                product_code="P001",
                document_id=12,
                document_type="manual",
            ),
            allowed_knowledge_base_ids={7},
        )

    assert exc_info.value.code == PRODUCT_ERROR_DOCUMENT_LINK
    assert exc_info.value.message == "文档尚未解析成功"


def test_link_document_rejects_document_outside_allowed_knowledge_base() -> None:
    service = _product_service(FakeProductRepository())

    with pytest.raises(BusinessException) as exc_info:
        service.link_document(
            ProductDocumentLinkCreate(
                product_code="P001",
                document_id=11,
                document_type="manual",
            ),
            allowed_knowledge_base_ids={99},
        )

    assert exc_info.value.code == PRODUCT_ERROR_DOCUMENT_LINK


def test_link_document_rejects_primary_non_manual() -> None:
    with pytest.raises(ValueError, match="非 manual 类型不能设置为主说明书"):
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=11,
            document_type="policy",
            is_primary=True,
        )


def test_link_document_is_idempotent_for_same_product_and_document() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    payload = ProductDocumentLinkCreate(
        product_code="P001",
        document_id=11,
        document_type="manual",
        is_primary=True,
        manual_version="v1",
    )

    first = service.link_document(payload, allowed_knowledge_base_ids={7})
    second = service.link_document(payload, allowed_knowledge_base_ids={7})

    assert first.id == second.id
    assert len(repository.links) == 1
    assert second.is_primary is True
    assert repository.committed is True


def test_existing_link_returns_without_lock_flush_commit_or_rollback() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    existing = SimpleNamespace(
        id=1,
        product_id=1,
        document_id=11,
        document_type="manual",
        is_primary=False,
        manual_version="v1",
        source_url="https://example.test/v1.pdf",
    )
    repository.links.append(existing)

    link = service.link_document(
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=11,
            document_type="manual",
            is_primary=True,
            manual_version="v2",
            source_url="https://example.test/v2.pdf",
        ),
        allowed_knowledge_base_ids={7},
    )

    assert link is existing
    assert existing.is_primary is False
    assert existing.manual_version == "v1"
    assert existing.source_url == "https://example.test/v1.pdf"
    assert repository.calls == []


def test_switch_primary_manual_unsets_old_primary_in_same_service_operation() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    repository.documents[13] = SimpleNamespace(
        id=13,
        knowledge_base_id=7,
        deleted_at=None,
        parse_status="success",
    )

    old_link = service.link_document(
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=11,
            document_type="manual",
            is_primary=True,
        ),
        allowed_knowledge_base_ids={7},
    )
    new_link = service.link_document(
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=13,
            document_type="manual",
            is_primary=True,
        ),
        allowed_knowledge_base_ids={7},
    )

    assert old_link.is_primary is False
    assert new_link.is_primary is True
    assert repository.primary_unset_for == 1


def test_new_primary_manual_transaction_order() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    old_link = SimpleNamespace(
        id=1,
        product_id=1,
        document_id=11,
        document_type="manual",
        is_primary=True,
        manual_version="v1",
        source_url=None,
    )
    repository.links.append(old_link)
    repository.documents[13] = SimpleNamespace(
        id=13,
        knowledge_base_id=7,
        deleted_at=None,
        parse_status="success",
    )

    service.link_document(
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=13,
            document_type="manual",
            is_primary=True,
        ),
        allowed_knowledge_base_ids={7},
    )

    assert repository.calls == [
        "lock_product",
        "begin_nested",
        "create_non_primary",
        "savepoint_release",
        "lock_old_primary",
        "unset_old",
        "flush",
        "set_new_primary",
        "flush",
        "commit",
    ]
    assert repository.links[-1].is_primary is True


def test_new_primary_manual_locks_product_even_without_old_primary() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    repository.documents[13] = SimpleNamespace(
        id=13,
        knowledge_base_id=7,
        deleted_at=None,
        parse_status="success",
    )

    service.link_document(
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=13,
            document_type="manual",
            is_primary=True,
        ),
        allowed_knowledge_base_ids={7},
    )

    assert repository.calls == [
        "lock_product",
        "begin_nested",
        "create_non_primary",
        "savepoint_release",
        "lock_old_primary",
        "set_new_primary",
        "flush",
        "commit",
    ]
    assert repository.links[-1].is_primary is True


def test_new_primary_promotion_flush_failure_rolls_back_transaction() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    repository.links.append(
        SimpleNamespace(
            id=1,
            product_id=1,
            document_id=11,
            document_type="manual",
            is_primary=True,
        )
    )
    repository.documents[13] = SimpleNamespace(
        id=13,
        knowledge_base_id=7,
        deleted_at=None,
        parse_status="success",
    )
    flush_count = {"count": 0}

    def failing_flush():
        flush_count["count"] += 1
        repository.calls.append("flush")
        if flush_count["count"] == 2:
            raise RuntimeError("promotion failed")

    repository.flush = failing_flush

    with pytest.raises(RuntimeError):
        service.link_document(
            ProductDocumentLinkCreate(
                product_code="P001",
                document_id=13,
                document_type="manual",
                is_primary=True,
            ),
            allowed_knowledge_base_ids={7},
        )

    assert repository.rolled_back is True
    assert "rollback" in repository.calls


def test_concurrent_unique_conflict_returns_existing_link_after_safe_rollback() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    existing_after_conflict = SimpleNamespace(
        id=21,
        product_id=1,
        document_id=13,
        document_type="manual",
        is_primary=False,
        manual_version="v1",
        source_url=None,
    )
    repository.documents[13] = SimpleNamespace(
        id=13,
        knowledge_base_id=7,
        deleted_at=None,
        parse_status="success",
    )
    repository.raise_on_create = _integrity_error(
        "uq_product_document_links_product_document"
    )

    original_get = repository.get_document_link
    call_count = {"count": 0}

    def get_document_link(product_id: int, document_id: int):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return None
        return existing_after_conflict

    repository.get_document_link = get_document_link

    old_primary = SimpleNamespace(
        id=1,
        product_id=1,
        document_id=11,
        document_type="manual",
        is_primary=True,
        manual_version="v1",
        source_url=None,
    )
    repository.links.append(old_primary)

    link = service.link_document(
        ProductDocumentLinkCreate(
            product_code="P001",
            document_id=13,
            document_type="manual",
            is_primary=True,
        ),
        allowed_knowledge_base_ids={7},
    )

    assert link is existing_after_conflict
    assert old_primary.is_primary is True
    assert existing_after_conflict.is_primary is False
    assert "savepoint_rollback" in repository.calls
    assert "rollback" not in repository.calls
    assert "lock_old_primary" not in repository.calls
    assert "unset_old" not in repository.calls
    assert "set_new_primary" not in repository.calls
    assert "commit" not in repository.calls
    repository.get_document_link = original_get


def test_other_integrity_error_is_not_swallowed() -> None:
    repository = FakeProductRepository()
    service = _product_service(repository)
    repository.documents[13] = SimpleNamespace(
        id=13,
        knowledge_base_id=7,
        deleted_at=None,
        parse_status="success",
    )
    repository.raise_on_create = _integrity_error("uq_product_document_links_primary_manual")

    with pytest.raises(IntegrityError):
        service.link_document(
            ProductDocumentLinkCreate(
                product_code="P001",
                document_id=13,
                document_type="manual",
                is_primary=False,
            ),
            allowed_knowledge_base_ids={7},
        )
    assert "rollback" in repository.calls


def test_product_repository_keyword_sql_includes_tags_as_bound_parameter() -> None:
    repository = ProductRepository(cast(Any, None))
    filters = ProductListFilters(keyword="宿舍")

    sql = repository.compile_list_sql_for_dialect(
        filters,
        dialect=postgresql.dialect(),
    )

    assert "jsonb_array_elements_text(products.tags)" in sql
    assert "ILIKE" in sql
    assert "宿舍" not in sql
    assert "%(product_code_1)s" in sql or "%(value_1)s" in sql


def test_product_repository_jsonb_contains_sql_uses_jsonb_contains_operator() -> None:
    repository = ProductRepository(cast(Any, None))
    filters = ProductListFilters(
        required_features=["易清洗"],
        excluded_features=["高噪音"],
        required_use_cases=["宿舍"],
    )

    sql = repository.compile_list_sql_for_dialect(
        filters,
        dialect=postgresql.dialect(),
    )

    assert "products.features @>" in sql
    assert "products.use_cases @>" in sql
    assert "NOT" in sql


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
        specifications={},
        tags=[],
        popularity_score=popularity_score,
        official_product_url=None,
        source_checked_at=None,
        is_active=True,
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )


def _product_service(repository: FakeProductRepository) -> ProductService:
    return ProductService(cast(ProductRepository, repository))


def _integrity_error(constraint_name: str) -> IntegrityError:
    class Diag:
        constraint_name: str

        def __init__(self, constraint_name: str) -> None:
            self.constraint_name = constraint_name

    class Orig(Exception):
        diag: Diag

        def __init__(self, constraint_name: str) -> None:
            self.diag = Diag(constraint_name)

        def __str__(self) -> str:
            return self.diag.constraint_name

    return IntegrityError("statement", {}, Orig(constraint_name))
