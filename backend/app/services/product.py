from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.models import Product, ProductDocumentLink
from backend.app.repositories.product import ProductListFilters, ProductRepository
from backend.app.schemas.product import (
    DOCUMENT_TYPE_VALUES,
    ProductCreate,
    ProductDocumentLinkCreate,
    ProductQuery,
    ProductUpdate,
)
from backend.app.services.base import BaseService
from sqlalchemy.exc import IntegrityError

SORT_COLUMNS = {
    "popularity": Product.popularity_score,
    "price": Product.price,
    "stock_quantity": Product.stock_quantity,
    "created_at": Product.created_at,
    "updated_at": Product.updated_at,
}

PRODUCT_ERROR_INVALID_ARGUMENT = 40020
PRODUCT_ERROR_DUPLICATE_CODE = 40021
PRODUCT_ERROR_INVALID_SORT = 40022
PRODUCT_ERROR_DOCUMENT_LINK = 40023
PRODUCT_ERROR_NOT_FOUND = 40420
PRODUCT_ERROR_DOCUMENT_NOT_FOUND = 40421
PRODUCT_DOCUMENT_UNIQUE_CONSTRAINT = "uq_product_document_links_product_document"


@dataclass(slots=True)
class ProductRecommendation:
    product: Product
    score: float
    reasons: list[str]


class ProductService(BaseService[ProductRepository]):
    def create(self, data: ProductCreate) -> Product:
        logger.info(f"Create product started | product_code={data.product_code}")
        if self.repository.get_by_product_code(data.product_code, include_deleted=True):
            logger.warning(f"Product code duplicated | product_code={data.product_code}")
            raise BusinessException(PRODUCT_ERROR_DUPLICATE_CODE, "商品编码已存在")

        product = self.repository.create_product(data.model_dump())
        logger.info(f"Create product succeeded | product_id={product.id}")
        return product

    def get(self, id: int, *, include_deleted: bool = False) -> Product:
        logger.info(f"Get product started | product_id={id}")
        product = self.repository.get(id, include_deleted=include_deleted)
        if product is None:
            logger.warning(f"Product not found | product_id={id}")
            raise BusinessException(PRODUCT_ERROR_NOT_FOUND, "商品不存在")
        logger.info(f"Get product succeeded | product_id={id}")
        return product

    def get_by_product_code(
        self,
        product_code: str,
        *,
        include_deleted: bool = False,
    ) -> Product:
        logger.info(f"Get product by code started | product_code={product_code}")
        product = self.repository.get_by_product_code(
            product_code,
            include_deleted=include_deleted,
        )
        if product is None:
            logger.warning(f"Product not found | product_code={product_code}")
            raise BusinessException(PRODUCT_ERROR_NOT_FOUND, "商品不存在")
        logger.info(f"Get product by code succeeded | product_code={product_code}")
        return product

    def update(self, id: int, data: ProductUpdate) -> Product:
        logger.info(f"Update product started | product_id={id}")
        product = self.get(id)
        update_data = data.model_dump(exclude_unset=True)
        product = self.repository.update_product(product, update_data)
        logger.info(f"Update product succeeded | product_id={id}")
        return product

    def deactivate(self, id: int) -> None:
        logger.info(f"Deactivate product started | product_id={id}")
        product = self.get(id)
        self.repository.update_product(
            product,
            {
                "is_active": False,
                "deleted_at": datetime.utcnow(),
            },
        )
        logger.info(f"Deactivate product succeeded | product_id={id}")

    def rollback(self) -> None:
        self.repository.rollback()

    def list(self, query: ProductQuery) -> tuple[list[Product], int]:
        logger.info("List products started")
        normalized = self.normalize_query(query)
        sort_column = self._sort_column(normalized.sort_by)
        filters = self._build_filters(normalized)
        offset = (normalized.page - 1) * normalized.page_size
        products, total = self.repository.list_products(
            filters,
            offset=offset,
            limit=normalized.page_size,
            sort_column=sort_column,
            sort_order=normalized.sort_order,
        )
        logger.info(f"List products succeeded | total={total}")
        return products, total

    def recommend(self, query: ProductQuery) -> tuple[list[ProductRecommendation], str | None]:
        logger.info("Recommend products started")
        normalized = self.normalize_query(query)
        sort_column = self._sort_column("popularity")
        filters = self._build_filters(normalized)
        candidates, _ = self.repository.list_products(
            filters,
            offset=0,
            limit=100,
            sort_column=sort_column,
            sort_order="desc",
        )
        if not candidates:
            logger.info("Recommend products returned no candidates")
            return [], "没有符合硬条件的商品"

        recommendations = [
            ProductRecommendation(
                product=product,
                score=self._recommendation_score(product, normalized),
                reasons=self._recommendation_reasons(product, normalized),
            )
            for product in candidates
        ]
        recommendations.sort(
            key=lambda item: (
                -item.score,
                -item.product.popularity_score,
                item.product.product_code,
            )
        )
        logger.info(f"Recommend products succeeded | total={len(recommendations[:3])}")
        return recommendations[:3], None

    def normalize_query(self, query: ProductQuery) -> ProductQuery:
        data = query.model_dump()
        data["required_features"] = self._unique_texts(
            [
                *data.get("required_features", []),
                *data.get("features", []),
            ]
        )
        data["preferred_features"] = self._unique_texts(data.get("preferred_features", []))
        data["excluded_features"] = self._unique_texts(data.get("excluded_features", []))
        data["required_use_cases"] = self._unique_texts(data.get("required_use_cases", []))
        data["preferred_use_cases"] = self._unique_texts(
            [
                *data.get("preferred_use_cases", []),
                *data.get("use_cases", []),
            ]
        )
        data["features"] = []
        data["use_cases"] = []
        return ProductQuery(**data)

    def list_document_links(self, product_id: int) -> list[ProductDocumentLink]:
        self.get(product_id)
        return self.repository.list_document_links(product_id)

    def link_document(
        self,
        data: ProductDocumentLinkCreate,
        *,
        allowed_knowledge_base_ids: set[int],
    ) -> ProductDocumentLink:
        logger.info(
            "Link product document started | "
            f"product_code={data.product_code} document_id={data.document_id}",
        )
        if data.document_type not in DOCUMENT_TYPE_VALUES:
            raise BusinessException(PRODUCT_ERROR_DOCUMENT_LINK, "文档类型不合法")
        if data.is_primary and data.document_type != "manual":
            raise BusinessException(PRODUCT_ERROR_DOCUMENT_LINK, "非说明书不能设为主说明书")
        if not allowed_knowledge_base_ids:
            raise BusinessException(PRODUCT_ERROR_DOCUMENT_LINK, "允许的说明书知识库不能为空")

        product = self.get_by_product_code(data.product_code)
        if product.deleted_at is not None or not product.is_active:
            raise BusinessException(PRODUCT_ERROR_NOT_FOUND, "商品不存在或已停用")

        document = self.repository.get_active_document(data.document_id)
        if document is None:
            raise BusinessException(PRODUCT_ERROR_DOCUMENT_NOT_FOUND, "文档不存在")
        if document.parse_status != "success":
            raise BusinessException(PRODUCT_ERROR_DOCUMENT_LINK, "文档尚未解析成功")
        if document.knowledge_base_id not in allowed_knowledge_base_ids:
            raise BusinessException(PRODUCT_ERROR_DOCUMENT_LINK, "文档不属于允许的产品说明书知识库")

        existing = self.repository.get_document_link(product.id, data.document_id)
        if existing is not None:
            logger.info(f"Product document link already exists | link_id={existing.id}")
            return existing

        link_data = data.model_dump(exclude={"product_code"})
        link_data["product_id"] = product.id
        link_data["is_primary"] = False
        try:
            locked_product = self.repository.get_for_update(product.id)
            if locked_product is None:
                raise BusinessException(PRODUCT_ERROR_NOT_FOUND, "商品不存在或已停用")
            link, created = self._create_document_link_with_idempotency(
                product.id,
                data.document_id,
                link_data,
            )
            if not created:
                return link
            if data.is_primary:
                self._promote_primary_manual(product.id, link)
            self.repository.commit()
            self.repository.refresh(link)
        except IntegrityError:
            self.repository.rollback()
            logger.exception("Link product document integrity error")
            raise
        except Exception:
            self.repository.rollback()
            logger.exception("Link product document failed")
            raise

        logger.info(f"Link product document succeeded | link_id={link.id}")
        return link

    def _create_document_link_with_idempotency(
        self,
        product_id: int,
        document_id: int,
        link_data: dict[str, Any],
    ) -> tuple[ProductDocumentLink, bool]:
        try:
            with self.repository.begin_nested():
                link = self.repository.create_document_link(link_data, flush=True)
                return link, True
        except IntegrityError as exc:
            if not self._is_product_document_unique_conflict(exc):
                raise
            existing = self.repository.get_document_link(product_id, document_id)
            if existing is None:
                raise
            return existing, False

    def _promote_primary_manual(
        self,
        product_id: int,
        link: ProductDocumentLink,
    ) -> None:
        old_primary = self.repository.get_primary_manual_link_for_update(product_id)
        if old_primary is not None and old_primary.id != link.id:
            self.repository.unset_primary_manual_link(old_primary)
            self.repository.flush()
        self.repository.set_primary_manual_link(link)
        self.repository.flush()

    def _is_product_document_unique_conflict(self, exc: IntegrityError) -> bool:
        constraint_name = getattr(getattr(exc, "orig", None), "diag", None)
        if constraint_name is not None:
            name = getattr(constraint_name, "constraint_name", None)
            return name == PRODUCT_DOCUMENT_UNIQUE_CONSTRAINT
        message = str(exc.orig)
        return PRODUCT_DOCUMENT_UNIQUE_CONSTRAINT in message

    def _build_filters(self, query: ProductQuery) -> ProductListFilters:
        return ProductListFilters(
            keyword=self._clean_text(query.keyword),
            brand=self._clean_text(query.brand),
            category=self._clean_text(query.category),
            model=self._clean_text(query.model),
            price_min=float(query.price_min) if query.price_min is not None else None,
            price_max=float(query.price_max) if query.price_max is not None else None,
            required_features=query.required_features,
            excluded_features=query.excluded_features,
            required_use_cases=query.required_use_cases,
            in_stock_only=query.in_stock_only,
            sale_status=query.sale_status,
            is_active=True,
            include_deleted=False,
        )

    def _sort_column(self, sort_by: str):
        sort_column = SORT_COLUMNS.get(sort_by)
        if sort_column is None:
            raise BusinessException(PRODUCT_ERROR_INVALID_SORT, "排序字段不支持")
        return sort_column

    def _recommendation_score(self, product: Product, query: ProductQuery) -> float:
        popularity = self._clamp(float(product.popularity_score) / 100)
        feature_match = self._match_score(product.features, query.preferred_features)
        use_case_match = self._match_score(product.use_cases, query.preferred_use_cases)
        price_preference = self._price_score(product.price, query.price_min, query.price_max)
        stock_score = self._clamp(float(product.stock_quantity) / 100)
        score = (
            popularity * 0.35
            + feature_match * 0.25
            + use_case_match * 0.25
            + price_preference * 0.10
            + stock_score * 0.05
        )
        return self._clamp(round(score, 6))

    def _recommendation_reasons(self, product: Product, query: ProductQuery) -> list[str]:
        reasons: list[str] = []
        matched_features = sorted(set(product.features or []) & set(query.preferred_features))
        matched_use_cases = sorted(set(product.use_cases or []) & set(query.preferred_use_cases))
        if matched_features:
            reasons.append("匹配偏好功能：" + "、".join(matched_features))
        if matched_use_cases:
            reasons.append("匹配使用场景：" + "、".join(matched_use_cases))
        if product.popularity_score > 0:
            reasons.append(f"模拟热度 {product.popularity_score}/100")
        if product.stock_quantity > 0:
            reasons.append("当前模拟库存可用")
        if not reasons:
            reasons.append("满足全部硬条件")
        return reasons

    def _price_score(
        self,
        price: Decimal,
        price_min: Decimal | None,
        price_max: Decimal | None,
    ) -> float:
        value = float(price)
        if price_min is not None and price_max is not None:
            min_value = float(price_min)
            max_value = float(price_max)
            if max_value == min_value:
                return 1.0 if value == min_value else 0.0
            return self._clamp(1 - ((value - min_value) / (max_value - min_value)))
        if price_max is not None and float(price_max) > 0:
            return self._clamp(1 - (value / float(price_max)))
        return 0.5

    def _match_score(self, values: list[str] | None, preferred_values: list[str]) -> float:
        if not preferred_values:
            return 1.0
        value_set = set(values or [])
        if not value_set:
            return 0.0
        return self._clamp(len(value_set & set(preferred_values)) / len(preferred_values))

    def _unique_texts(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = self._clean_text(value)
            if cleaned and cleaned not in seen:
                result.append(cleaned)
                seen.add(cleaned)
        return result

    def _clean_text(self, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
