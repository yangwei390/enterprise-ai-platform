from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SALE_STATUS_VALUES = {"on_sale", "off_sale", "pre_sale", "discontinued"}
DOCUMENT_TYPE_VALUES = {"manual", "warranty", "policy", "other"}
SORT_BY_VALUES = {"popularity", "price", "stock_quantity", "created_at", "updated_at"}
SORT_ORDER_VALUES = {"asc", "desc"}


class ProductBase(BaseModel):
    brand: str
    name: str
    model: str
    category: str
    description: str | None = None
    price: Decimal
    currency: str = "CNY"
    stock_quantity: int = 0
    sale_status: str = "on_sale"
    features: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    specifications: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    popularity_score: int = 0
    official_product_url: str | None = None
    source_checked_at: datetime | None = None
    is_active: bool = True

    @field_validator("brand", "name", "model", "category")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("字段不能为空")
        return value.strip()

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency 必须是三字符大写货币代码")
        return normalized

    @field_validator("sale_status")
    @classmethod
    def validate_sale_status(cls, value: str) -> str:
        if value not in SALE_STATUS_VALUES:
            raise ValueError("sale_status 不合法")
        return value

    @field_validator("price")
    @classmethod
    def validate_price(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("price 不能小于 0")
        return value

    @field_validator("stock_quantity")
    @classmethod
    def validate_stock_quantity(cls, value: int) -> int:
        if value < 0:
            raise ValueError("stock_quantity 不能小于 0")
        return value

    @field_validator("popularity_score")
    @classmethod
    def validate_popularity_score(cls, value: int) -> int:
        if value < 0 or value > 100:
            raise ValueError("popularity_score 必须在 0 到 100 之间")
        return value


class ProductCreate(ProductBase):
    product_code: str

    @field_validator("product_code")
    @classmethod
    def validate_product_code(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("product_code 不能为空")
        return value.strip()


class ProductUpdate(BaseModel):
    brand: str | None = None
    name: str | None = None
    model: str | None = None
    category: str | None = None
    description: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    stock_quantity: int | None = None
    sale_status: str | None = None
    features: list[str] | None = None
    use_cases: list[str] | None = None
    specifications: dict | None = None
    tags: list[str] | None = None
    popularity_score: int | None = None
    official_product_url: str | None = None
    source_checked_at: datetime | None = None
    is_active: bool | None = None

    @field_validator("brand", "name", "model", "category")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("字段不能为空")
        return value.strip()

    @field_validator("currency")
    @classmethod
    def validate_optional_currency(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency 必须是三字符大写货币代码")
        return normalized

    @field_validator("sale_status")
    @classmethod
    def validate_optional_sale_status(cls, value: str | None) -> str | None:
        if value is not None and value not in SALE_STATUS_VALUES:
            raise ValueError("sale_status 不合法")
        return value

    @field_validator("price")
    @classmethod
    def validate_optional_price(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and value < 0:
            raise ValueError("price 不能小于 0")
        return value

    @field_validator("stock_quantity")
    @classmethod
    def validate_optional_stock_quantity(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("stock_quantity 不能小于 0")
        return value

    @field_validator("popularity_score")
    @classmethod
    def validate_optional_popularity_score(cls, value: int | None) -> int | None:
        if value is not None and (value < 0 or value > 100):
            raise ValueError("popularity_score 必须在 0 到 100 之间")
        return value


class ProductQuery(BaseModel):
    keyword: str | None = None
    brand: str | None = None
    category: str | None = None
    model: str | None = None
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    required_features: list[str] = Field(default_factory=list)
    excluded_features: list[str] = Field(default_factory=list)
    preferred_features: list[str] = Field(default_factory=list)
    required_use_cases: list[str] = Field(default_factory=list)
    preferred_use_cases: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    in_stock_only: bool = True
    sale_status: str | None = "on_sale"
    sort_by: str = "popularity"
    sort_order: str = "desc"
    page: int = 1
    page_size: int = 20

    @model_validator(mode="after")
    def validate_ranges(self) -> ProductQuery:
        if self.price_min is not None and self.price_min < 0:
            raise ValueError("price_min 不能小于 0")
        if self.price_max is not None and self.price_max < 0:
            raise ValueError("price_max 不能小于 0")
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min 不能大于 price_max")
        if self.sale_status is not None and self.sale_status not in SALE_STATUS_VALUES:
            raise ValueError("sale_status 不合法")
        if self.sort_by not in SORT_BY_VALUES:
            raise ValueError("sort_by 不在白名单中")
        if self.sort_order not in SORT_ORDER_VALUES:
            raise ValueError("sort_order 不合法")
        if self.page < 1:
            raise ValueError("page 必须大于等于 1")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("page_size 必须在 1 到 100 之间")
        return self


class ProductResponse(BaseModel):
    id: int
    product_code: str
    brand: str
    name: str
    model: str
    category: str
    description: str | None
    price: Decimal
    currency: str
    stock_quantity: int
    sale_status: str
    features: list[str]
    use_cases: list[str]
    specifications: dict
    tags: list[str]
    popularity_score: int
    official_product_url: str | None
    source_checked_at: datetime | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class ProductListResponse(BaseModel):
    items: list[ProductResponse]
    total: int
    page: int
    page_size: int


class ProductDocumentLinkCreate(BaseModel):
    product_code: str
    document_id: int
    document_type: str = "manual"
    is_primary: bool = False
    manual_version: str | None = None
    source_url: str | None = None
    downloaded_at: datetime | None = None

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, value: str) -> str:
        if value not in DOCUMENT_TYPE_VALUES:
            raise ValueError("document_type 不合法")
        return value

    @model_validator(mode="after")
    def validate_primary_type(self) -> ProductDocumentLinkCreate:
        if self.is_primary and self.document_type != "manual":
            raise ValueError("非 manual 类型不能设置为主说明书")
        return self


class ProductDocumentLinkResponse(BaseModel):
    id: int
    product_id: int
    document_id: int
    document_type: str
    is_primary: bool
    manual_version: str | None
    source_url: str | None
    downloaded_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProductDocumentLinkListResponse(BaseModel):
    items: list[ProductDocumentLinkResponse]
    total: int


class ProductRecommendationItem(BaseModel):
    product: ProductResponse
    score: float
    reasons: list[str]


class ProductRecommendationResponse(BaseModel):
    items: list[ProductRecommendationItem]
    total: int
    no_result_reason: str | None = None
