from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from backend.app.models.base import Base
from backend.app.models.mixins import SoftDeleteMixin, TimestampMixin
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .product_document_link import ProductDocumentLink


class Product(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("price >= 0", name="ck_products_price_non_negative"),
        CheckConstraint(
            "stock_quantity >= 0",
            name="ck_products_stock_quantity_non_negative",
        ),
        CheckConstraint(
            "popularity_score >= 0 AND popularity_score <= 100",
            name="ck_products_popularity_score_range",
        ),
        CheckConstraint(
            "sale_status IN ('on_sale', 'off_sale', 'pre_sale', 'discontinued')",
            name="ck_products_sale_status",
        ),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_products_currency_code"),
        UniqueConstraint("product_code", name="uq_products_product_code"),
        Index("ix_products_brand", "brand"),
        Index("ix_products_category", "category"),
        Index("ix_products_model", "model"),
        Index("ix_products_sale_status", "sale_status"),
        Index("ix_products_is_active", "is_active"),
        Index("ix_products_price", "price"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    product_code: Mapped[str] = mapped_column(String(64), nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3),
        default="CNY",
        server_default=text("'CNY'"),
        nullable=False,
    )
    stock_quantity: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    sale_status: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        server_default=text("'[]'::jsonb"),
        nullable=False,
    )
    use_cases: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        server_default=text("'[]'::jsonb"),
        nullable=False,
    )
    specifications: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    tags: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        server_default=text("'[]'::jsonb"),
        nullable=False,
    )
    popularity_score: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    official_product_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
    )

    document_links: Mapped[list[ProductDocumentLink]] = relationship(
        "ProductDocumentLink",
        back_populates="product",
        cascade="all, delete-orphan",
    )
