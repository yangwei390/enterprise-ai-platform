"""create product catalog tables

Revision ID: 6d4e9a2f1b8c
Revises: 8f2a1c4d9b7e
Create Date: 2026-07-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "6d4e9a2f1b8c"
down_revision: str | Sequence[str] | None = "8f2a1c4d9b7e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("product_code", sa.String(length=64), nullable=False),
        sa.Column("brand", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column(
            "currency",
            sa.String(length=3),
            server_default=sa.text("'CNY'"),
            nullable=False,
        ),
        sa.Column(
            "stock_quantity",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("sale_status", sa.String(length=32), nullable=False),
        sa.Column(
            "features",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "use_cases",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "specifications",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "popularity_score",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("official_product_url", sa.String(length=1024), nullable=True),
        sa.Column("source_checked_at", sa.DateTime(), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="ck_products_currency_code"),
        sa.CheckConstraint(
            "popularity_score >= 0 AND popularity_score <= 100",
            name="ck_products_popularity_score_range",
        ),
        sa.CheckConstraint("price >= 0", name="ck_products_price_non_negative"),
        sa.CheckConstraint(
            "sale_status IN ('on_sale', 'off_sale', 'pre_sale', 'discontinued')",
            name="ck_products_sale_status",
        ),
        sa.CheckConstraint(
            "stock_quantity >= 0",
            name="ck_products_stock_quantity_non_negative",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_code", name="uq_products_product_code"),
    )
    op.create_index("ix_products_brand", "products", ["brand"], unique=False)
    op.create_index("ix_products_category", "products", ["category"], unique=False)
    op.create_index("ix_products_is_active", "products", ["is_active"], unique=False)
    op.create_index("ix_products_model", "products", ["model"], unique=False)
    op.create_index("ix_products_price", "products", ["price"], unique=False)
    op.create_index("ix_products_sale_status", "products", ["sale_status"], unique=False)

    op.create_table(
        "product_document_links",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("document_type", sa.String(length=32), nullable=False),
        sa.Column(
            "is_primary",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("manual_version", sa.String(length=128), nullable=True),
        sa.Column("source_url", sa.String(length=1024), nullable=True),
        sa.Column("downloaded_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "document_type IN ('manual', 'warranty', 'policy', 'other')",
            name="ck_product_document_links_document_type",
        ),
        sa.CheckConstraint(
            "document_type = 'manual' OR is_primary = false",
            name="ck_product_document_links_primary_manual_only",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "product_id",
            "document_id",
            name="uq_product_document_links_product_document",
        ),
    )
    op.create_index(
        "uq_product_document_links_primary_manual",
        "product_document_links",
        ["product_id"],
        unique=True,
        postgresql_where=sa.text("is_primary = true AND document_type = 'manual'"),
    )
    op.create_index(
        "ix_product_document_links_document_id",
        "product_document_links",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_product_document_links_document_id",
        table_name="product_document_links",
    )
    op.drop_index(
        "uq_product_document_links_primary_manual",
        table_name="product_document_links",
        postgresql_where=sa.text("is_primary = true AND document_type = 'manual'"),
    )
    op.drop_table("product_document_links")
    op.drop_index("ix_products_sale_status", table_name="products")
    op.drop_index("ix_products_price", table_name="products")
    op.drop_index("ix_products_model", table_name="products")
    op.drop_index("ix_products_is_active", table_name="products")
    op.drop_index("ix_products_category", table_name="products")
    op.drop_index("ix_products_brand", table_name="products")
    op.drop_table("products")
