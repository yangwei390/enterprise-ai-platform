from __future__ import annotations

from decimal import Decimal
from typing import Literal

from backend.app.db.session import get_db
from backend.app.repositories.product import ProductRepository
from backend.app.schemas import ApiResponse, success
from backend.app.schemas.product import (
    ProductCreate,
    ProductDocumentLinkListResponse,
    ProductDocumentLinkResponse,
    ProductListResponse,
    ProductQuery,
    ProductRecommendationItem,
    ProductRecommendationResponse,
    ProductResponse,
    ProductUpdate,
)
from backend.app.services.product import ProductService
from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.orm import Session

router = APIRouter()


def build_product_query(
    keyword: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    model: str | None = None,
    price_min: Decimal | None = None,
    price_max: Decimal | None = None,
    required_features: list[str] = Query(default_factory=list),
    excluded_features: list[str] = Query(default_factory=list),
    preferred_features: list[str] = Query(default_factory=list),
    required_use_cases: list[str] = Query(default_factory=list),
    preferred_use_cases: list[str] = Query(default_factory=list),
    features: list[str] = Query(default_factory=list),
    use_cases: list[str] = Query(default_factory=list),
    in_stock_only: bool = True,
    sale_status: str | None = "on_sale",
    sort_by: Literal["popularity", "price", "stock_quantity", "created_at", "updated_at"] = (
        "popularity"
    ),
    sort_order: Literal["asc", "desc"] = "desc",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> ProductQuery:
    try:
        return ProductQuery(
            keyword=keyword,
            brand=brand,
            category=category,
            model=model,
            price_min=price_min,
            price_max=price_max,
            required_features=required_features,
            excluded_features=excluded_features,
            preferred_features=preferred_features,
            required_use_cases=required_use_cases,
            preferred_use_cases=preferred_use_cases,
            features=features,
            use_cases=use_cases,
            in_stock_only=in_stock_only,
            sale_status=sale_status,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def get_product_service(db: Session = Depends(get_db)) -> ProductService:
    repository = ProductRepository(db)
    return ProductService(repository)


@router.get("/products", response_model=ApiResponse)
def list_products(
    query: ProductQuery = Depends(build_product_query),
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    products, total = service.list(query)
    items = [ProductResponse.model_validate(product) for product in products]
    return success(
        data=ProductListResponse(
            items=items,
            total=total,
            page=query.page,
            page_size=query.page_size,
        )
    )


@router.get("/products/recommendations", response_model=ApiResponse)
def recommend_products(
    query: ProductQuery = Depends(build_product_query),
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    recommendations, no_result_reason = service.recommend(query)
    items = [
        ProductRecommendationItem(
            product=ProductResponse.model_validate(item.product),
            score=item.score,
            reasons=item.reasons,
        )
        for item in recommendations
    ]
    return success(
        data=ProductRecommendationResponse(
            items=items,
            total=len(items),
            no_result_reason=no_result_reason,
        )
    )


@router.post("/products", response_model=ApiResponse)
def create_product(
    data: ProductCreate,
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    product = service.create(data)
    return success(data=ProductResponse.model_validate(product))


@router.get("/products/{id}", response_model=ApiResponse)
def get_product(
    id: int,
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    product = service.get(id)
    return success(data=ProductResponse.model_validate(product))


@router.put("/products/{id}", response_model=ApiResponse)
def update_product(
    id: int,
    data: ProductUpdate,
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    product = service.update(id, data)
    return success(data=ProductResponse.model_validate(product))


@router.delete("/products/{id}", response_model=ApiResponse)
def delete_product(
    id: int,
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    service.deactivate(id)
    return success(data={"deleted": True})


@router.get("/products/{id}/documents", response_model=ApiResponse)
def list_product_documents(
    id: int,
    service: ProductService = Depends(get_product_service),
) -> ApiResponse:
    links = service.list_document_links(id)
    items = [ProductDocumentLinkResponse.model_validate(link) for link in links]
    return success(data=ProductDocumentLinkListResponse(items=items, total=len(items)))
