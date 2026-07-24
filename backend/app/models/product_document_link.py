from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from backend.app.models.base import Base
from backend.app.models.mixins import TimestampMixin
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .document import Document
    from .product import Product


class ProductDocumentLink(TimestampMixin, Base):
    __tablename__ = "product_document_links"
    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "document_id",
            name="uq_product_document_links_product_document",
        ),
        CheckConstraint(
            "document_type IN ('manual', 'warranty', 'policy', 'other')",
            name="ck_product_document_links_document_type",
        ),
        CheckConstraint(
            "document_type = 'manual' OR is_primary = false",
            name="ck_product_document_links_primary_manual_only",
        ),
        Index(
            "uq_product_document_links_primary_manual",
            "product_id",
            unique=True,
            postgresql_where=text("is_primary = true AND document_type = 'manual'"),
        ),
        Index("ix_product_document_links_document_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
    )
    manual_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    product: Mapped[Product] = relationship("Product", back_populates="document_links")
    document: Mapped[Document] = relationship("Document")
