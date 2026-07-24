import importlib.util
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from backend.app.models import Base, Product, ProductDocumentLink
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB


def test_product_tables_are_registered_in_metadata() -> None:
    assert "products" in Base.metadata.tables
    assert "product_document_links" in Base.metadata.tables


def test_product_columns_types_defaults_and_nullable_flags() -> None:
    table = Product.__table__

    assert isinstance(table.c.product_code.type, String)
    assert table.c.product_code.nullable is False
    assert table.c.brand.nullable is False
    assert table.c.name.nullable is False
    assert table.c.model.nullable is False
    assert table.c.category.nullable is False
    assert table.c.price.nullable is False
    assert table.c.price.type.precision == 12
    assert table.c.price.type.scale == 2
    assert table.c.currency.nullable is False
    assert table.c.currency.default.arg == "CNY"
    assert str(table.c.currency.server_default.arg) == "'CNY'"
    assert table.c.stock_quantity.default.arg == 0
    assert str(table.c.stock_quantity.server_default.arg) == "0"
    assert table.c.popularity_score.default.arg == 0
    assert str(table.c.popularity_score.server_default.arg) == "0"
    assert table.c.is_active.default.arg is True
    assert str(table.c.is_active.server_default.arg) == "true"

    assert isinstance(table.c.features.type, JSONB)
    assert table.c.features.default.arg.__name__ == "list"
    assert isinstance(table.c.use_cases.type, JSONB)
    assert table.c.use_cases.default.arg.__name__ == "list"
    assert isinstance(table.c.specifications.type, JSONB)
    assert table.c.specifications.default.arg.__name__ == "dict"
    assert isinstance(table.c.tags.type, JSONB)
    assert table.c.tags.default.arg.__name__ == "list"

    product = Product(
        product_code="P001",
        brand="Brand",
        name="Name",
        model="M1",
        category="Category",
        price=Decimal("199.00"),
        sale_status="on_sale",
    )
    assert product.features is None
    assert product.use_cases is None


def test_product_constraints_and_indexes() -> None:
    table = Product.__table__
    unique_constraints = _constraints(table, UniqueConstraint)
    check_constraints = _constraints(table, CheckConstraint)
    indexes = {index.name: index for index in table.indexes}

    assert any(
        ("product_code",) == tuple(item.name for item in constraint.columns)
        and constraint.name == "uq_products_product_code"
        for constraint in unique_constraints
    )
    assert _check_sql(check_constraints, "price >= 0")
    assert _check_sql(check_constraints, "stock_quantity >= 0")
    assert _check_sql(check_constraints, "popularity_score >= 0")
    assert _check_sql(check_constraints, "popularity_score <= 100")
    assert _check_sql(check_constraints, "sale_status IN")
    assert _check_sql(check_constraints, "currency ~")

    assert set(indexes) >= {
        "ix_products_brand",
        "ix_products_category",
        "ix_products_model",
        "ix_products_sale_status",
        "ix_products_is_active",
        "ix_products_price",
    }
    assert tuple(indexes["ix_products_price"].columns.keys()) == ("price",)


def test_product_document_link_columns_defaults_and_foreign_keys() -> None:
    table = ProductDocumentLink.__table__

    assert table.c.product_id.nullable is False
    assert table.c.document_id.nullable is False
    assert table.c.document_type.nullable is False
    assert table.c.is_primary.nullable is False
    assert table.c.is_primary.default.arg is False
    assert str(table.c.is_primary.server_default.arg) == "false"

    foreign_keys = _constraints(table, ForeignKeyConstraint)
    product_fk = _foreign_key(foreign_keys, "product_id")
    document_fk = _foreign_key(foreign_keys, "document_id")

    assert product_fk.referred_table.name == "products"
    assert product_fk.ondelete == "CASCADE"
    assert document_fk.referred_table.name == "documents"
    assert document_fk.ondelete == "CASCADE"


def test_product_document_link_constraints_and_partial_unique_index() -> None:
    table = ProductDocumentLink.__table__
    unique_constraints = _constraints(table, UniqueConstraint)
    check_constraints = _constraints(table, CheckConstraint)
    indexes = {index.name: index for index in table.indexes}

    assert any(
        ("product_id", "document_id")
        == tuple(item.name for item in constraint.columns)
        for constraint in unique_constraints
    )
    assert _check_sql(check_constraints, "document_type IN")
    assert _check_sql(check_constraints, "document_type = 'manual' OR is_primary = false")

    primary_manual_index = indexes["uq_product_document_links_primary_manual"]
    assert primary_manual_index.unique is True
    assert tuple(primary_manual_index.columns.keys()) == ("product_id",)
    where_clause = str(primary_manual_index.dialect_options["postgresql"]["where"])
    assert "is_primary = true" in where_clause
    assert "document_type = 'manual'" in where_clause

    document_id_index = indexes["ix_product_document_links_document_id"]
    assert document_id_index.unique is False
    assert tuple(document_id_index.columns.keys()) == ("document_id",)


def test_product_catalog_migration_revision_chain() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "6d4e9a2f1b8c_create_product_catalog_tables.py"
    )
    spec = importlib.util.spec_from_file_location("product_catalog_migration", migration_path)
    assert spec is not None
    assert spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    assert migration.revision == "6d4e9a2f1b8c"
    assert migration.down_revision == "8f2a1c4d9b7e"


def test_product_catalog_migration_offline_sql_contains_expected_schema() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "upgrade",
            "8f2a1c4d9b7e:6d4e9a2f1b8c",
            "--sql",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    sql = result.stdout

    assert "CREATE TABLE products" in sql
    assert "CREATE TABLE product_document_links" in sql
    assert "ck_products_currency_code" in sql
    assert "ck_products_popularity_score_range" in sql
    assert "ck_products_price_non_negative" in sql
    assert "ck_products_sale_status" in sql
    assert "ck_products_stock_quantity_non_negative" in sql
    assert "ck_product_document_links_document_type" in sql
    assert "ck_product_document_links_primary_manual_only" in sql
    assert "uq_products_product_code" in sql
    assert "uq_product_document_links_product_document" in sql
    assert "uq_product_document_links_primary_manual" in sql
    assert "WHERE is_primary = true AND document_type = 'manual'" in sql
    assert "ix_product_document_links_document_id" in sql


def _constraints(table, constraint_type):
    return [
        constraint
        for constraint in table.constraints
        if isinstance(constraint, constraint_type)
    ]


def _check_sql(constraints: list[CheckConstraint], fragment: str) -> bool:
    return any(fragment in str(constraint.sqltext) for constraint in constraints)


def _foreign_key(
    constraints: list[ForeignKeyConstraint],
    local_column_name: str,
) -> ForeignKeyConstraint:
    for constraint in constraints:
        columns = [column.name for column in constraint.columns]
        if columns == [local_column_name]:
            return constraint
    raise AssertionError(f"foreign key not found: {local_column_name}")
