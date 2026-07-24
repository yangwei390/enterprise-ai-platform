from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.models import Document, Product, ProductDocumentLink
from backend.app.repositories.base import BaseRepository
from sqlalchemy import Select, exists, func, not_, or_, select
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import InstrumentedAttribute, selectinload
from sqlalchemy.sql import Select as SqlSelect


@dataclass(slots=True)
class ProductListFilters:
    keyword: str | None = None
    brand: str | None = None
    category: str | None = None
    model: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    required_features: list[str] = field(default_factory=list)
    excluded_features: list[str] = field(default_factory=list)
    required_use_cases: list[str] = field(default_factory=list)
    in_stock_only: bool | None = None
    sale_status: str | None = None
    is_active: bool | None = True
    include_deleted: bool = False


class ProductRepository(BaseRepository):
    PRODUCT_DOCUMENT_UNIQUE_CONSTRAINT = "uq_product_document_links_product_document"

    def get(self, id: int, *, include_deleted: bool = False) -> Product | None:
        statement = select(Product).where(Product.id == id)
        if not include_deleted:
            statement = statement.where(Product.deleted_at.is_(None))
        return self.db.execute(statement).scalar_one_or_none()

    def get_for_update(self, id: int) -> Product | None:
        statement = (
            select(Product)
            .where(
                Product.id == id,
                Product.deleted_at.is_(None),
            )
            .with_for_update()
        )
        return self.db.execute(statement).scalar_one_or_none()

    def get_by_product_code(
        self,
        product_code: str,
        *,
        include_deleted: bool = False,
    ) -> Product | None:
        statement = select(Product).where(Product.product_code == product_code)
        if not include_deleted:
            statement = statement.where(Product.deleted_at.is_(None))
        return self.db.execute(statement).scalar_one_or_none()

    def get_by_model(self, model: str, *, include_deleted: bool = False) -> Product | None:
        statement = select(Product).where(Product.model == model)
        if not include_deleted:
            statement = statement.where(Product.deleted_at.is_(None))
        return self.db.execute(statement).scalar_one_or_none()

    def create_product(self, data: dict[str, Any], *, commit: bool = True) -> Product:
        product = Product(**data)
        self.db.add(product)
        if commit:
            self.db.commit()
            self.db.refresh(product)
        else:
            self.db.flush()
        return product

    def update_product(
        self,
        product: Product,
        data: dict[str, Any],
        *,
        commit: bool = True,
    ) -> Product:
        for field_name, value in data.items():
            setattr(product, field_name, value)
        if commit:
            self.db.commit()
            self.db.refresh(product)
        else:
            self.db.flush()
        return product

    def list_products(
        self,
        filters: ProductListFilters,
        *,
        offset: int,
        limit: int,
        sort_column: InstrumentedAttribute,
        sort_order: str,
    ) -> tuple[list[Product], int]:
        statement = self._apply_filters(select(Product), filters)
        count_statement = self._apply_filters(select(func.count(Product.id)), filters)

        order_expression = (
            sort_column.asc() if sort_order == "asc" else sort_column.desc()
        )
        statement = statement.order_by(order_expression, Product.product_code.asc())
        statement = statement.offset(offset).limit(limit)

        products = list(self.db.execute(statement).scalars().all())
        total = self.db.execute(count_statement).scalar_one()
        return products, total

    def list_document_links(self, product_id: int) -> list[ProductDocumentLink]:
        statement = (
            select(ProductDocumentLink)
            .where(ProductDocumentLink.product_id == product_id)
            .options(
                selectinload(ProductDocumentLink.document),
            )
            .order_by(ProductDocumentLink.is_primary.desc(), ProductDocumentLink.id.asc())
        )
        return list(self.db.execute(statement).scalars().all())

    def get_document_link(
        self,
        product_id: int,
        document_id: int,
    ) -> ProductDocumentLink | None:
        statement = select(ProductDocumentLink).where(
            ProductDocumentLink.product_id == product_id,
            ProductDocumentLink.document_id == document_id,
        )
        return self.db.execute(statement).scalar_one_or_none()

    def create_document_link(
        self,
        data: dict[str, Any],
        *,
        flush: bool = True,
    ) -> ProductDocumentLink:
        link = ProductDocumentLink(**data)
        self.db.add(link)
        if flush:
            self.db.flush()
        return link

    def get_primary_manual_link(self, product_id: int) -> ProductDocumentLink | None:
        statement = select(ProductDocumentLink).where(
            ProductDocumentLink.product_id == product_id,
            ProductDocumentLink.document_type == "manual",
            ProductDocumentLink.is_primary.is_(True),
        )
        return self.db.execute(statement).scalar_one_or_none()

    def get_primary_manual_link_for_update(
        self,
        product_id: int,
    ) -> ProductDocumentLink | None:
        statement = select(ProductDocumentLink).where(
            ProductDocumentLink.product_id == product_id,
            ProductDocumentLink.document_type == "manual",
            ProductDocumentLink.is_primary.is_(True),
        ).with_for_update()
        return self.db.execute(statement).scalar_one_or_none()

    def unset_primary_manual_link(self, link: ProductDocumentLink) -> None:
        link.is_primary = False

    def set_primary_manual_link(self, link: ProductDocumentLink) -> None:
        link.is_primary = True

    def get_active_document(self, document_id: int) -> Document | None:
        statement = select(Document).where(
            Document.id == document_id,
            Document.deleted_at.is_(None),
        )
        return self.db.execute(statement).scalar_one_or_none()

    def commit(self) -> None:
        self.db.commit()

    def flush(self) -> None:
        self.db.flush()

    def rollback(self) -> None:
        self.db.rollback()

    def begin_nested(self):
        return self.db.begin_nested()

    def refresh(self, obj: Any) -> None:
        self.db.refresh(obj)

    def _apply_filters(
        self,
        statement: Select[tuple[Product]] | Select[tuple[int]],
        filters: ProductListFilters,
    ) -> Select[tuple[Product]] | Select[tuple[int]]:
        if not filters.include_deleted:
            statement = statement.where(Product.deleted_at.is_(None))
        if filters.is_active is not None:
            statement = statement.where(Product.is_active.is_(filters.is_active))
        if filters.keyword:
            keyword = f"%{filters.keyword}%"
            statement = statement.where(
                or_(
                    Product.product_code.ilike(keyword),
                    Product.name.ilike(keyword),
                    Product.model.ilike(keyword),
                    Product.description.ilike(keyword),
                    self._tags_keyword_exists(keyword),
                )
            )
        if filters.brand:
            statement = statement.where(Product.brand == filters.brand)
        if filters.category:
            statement = statement.where(Product.category == filters.category)
        if filters.model:
            statement = statement.where(Product.model == filters.model)
        if filters.price_min is not None:
            statement = statement.where(Product.price >= filters.price_min)
        if filters.price_max is not None:
            statement = statement.where(Product.price <= filters.price_max)
        if filters.sale_status:
            statement = statement.where(Product.sale_status == filters.sale_status)
        if filters.in_stock_only:
            statement = statement.where(Product.stock_quantity > 0)
        for feature in filters.required_features:
            statement = statement.where(Product.features.contains([feature]))
        for feature in filters.excluded_features:
            statement = statement.where(not_(Product.features.contains([feature])))
        for use_case in filters.required_use_cases:
            statement = statement.where(Product.use_cases.contains([use_case]))
        return statement

    def _tags_keyword_exists(self, keyword: str):
        tag_values = (
            func.jsonb_array_elements_text(Product.tags)
            .table_valued("value")
            .alias("tag_values")
        )
        return exists(
            select(1).select_from(tag_values).where(tag_values.c.value.ilike(keyword))
        )

    def compile_list_sql_for_dialect(
        self,
        filters: ProductListFilters,
        *,
        dialect: Dialect,
    ) -> str:
        statement: SqlSelect = self._apply_filters(select(Product), filters)
        return str(statement.compile(dialect=dialect))
