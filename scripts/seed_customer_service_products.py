from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config.settings import settings  # noqa: E402
from backend.app.db.session import SessionLocal  # noqa: E402
from backend.app.exceptions import BusinessException  # noqa: E402
from backend.app.repositories.product import ProductRepository  # noqa: E402
from backend.app.schemas.product import ProductCreate, ProductUpdate  # noqa: E402
from backend.app.services.product import PRODUCT_ERROR_NOT_FOUND, ProductService  # noqa: E402
from pydantic import ValidationError  # noqa: E402


class ProductSeedService(Protocol):
    def get_by_product_code(
        self,
        product_code: str,
        *,
        include_deleted: bool = False,
    ) -> Any: ...

    def create(self, data: ProductCreate) -> Any: ...

    def update(self, id: int, data: ProductUpdate) -> Any: ...

    def rollback(self) -> Any: ...


@dataclass(slots=True)
class SeedStats:
    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0


def load_fixture(path: Path) -> list[ProductCreate]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["products"] if isinstance(raw, dict) and "products" in raw else raw
    if not isinstance(items, list):
        raise ValueError("商品 Fixture 必须是数组，或包含 products 数组")

    products: list[ProductCreate] = []
    for item in items:
        if _contains_key(item, "document_id"):
            raise ValueError("商品 Fixture 不允许包含 document_id")
        products.append(ProductCreate(**item))
    return products


def run_seed(
    fixture_path: Path,
    service: ProductSeedService,
    *,
    dry_run: bool = True,
) -> SeedStats:
    stats = SeedStats()
    products = load_fixture(fixture_path)
    for product_data in products:
        try:
            existing = service.get_by_product_code(product_data.product_code)
        except BusinessException as exc:
            if exc.code != PRODUCT_ERROR_NOT_FOUND:
                stats.failed += 1
                continue
            existing = None
        except Exception:
            stats.failed += 1
            continue

        if existing is None:
            if not dry_run:
                try:
                    service.create(product_data)
                except Exception as exc:
                    _rollback(service)
                    _print_item_error(product_data.product_code, exc)
                    stats.failed += 1
                    continue
            stats.created += 1
            continue

        update_payload = product_data.model_dump(exclude={"product_code"})
        if _matches_existing(existing, update_payload):
            stats.skipped += 1
            continue

        if not dry_run:
            try:
                service.update(existing.id, ProductUpdate(**update_payload))
            except Exception as exc:
                _rollback(service)
                _print_item_error(product_data.product_code, exc)
                stats.failed += 1
                continue
        stats.updated += 1
    return stats


def build_service() -> ProductService:
    db = SessionLocal()
    return ProductService(ProductRepository(db))


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed mock customer service products")
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="默认 dry-run；传入该参数才写入数据库",
    )
    args = parser.parse_args()

    dry_run = not args.apply
    print(_database_summary())
    print(f"mode={'dry-run' if dry_run else 'apply'}")

    service = build_service()
    try:
        stats = run_seed(args.fixture, service, dry_run=dry_run)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"created={stats.created} "
        f"updated={stats.updated} "
        f"skipped={stats.skipped} "
        f"failed={stats.failed}"
    )
    return 1 if stats.failed else 0


def _database_summary() -> str:
    return (
        "target_db="
        f"{settings.POSTGRES_USER}@{settings.POSTGRES_HOST}:"
        f"{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


def _rollback(service: ProductSeedService) -> None:
    rollback = getattr(service, "rollback", None)
    if callable(rollback):
        rollback()


def _print_item_error(product_code: str, exc: Exception) -> None:
    print(
        f"seed item failed | product_code={product_code} | error={type(exc).__name__}",
        file=sys.stderr,
    )


def _contains_key(value: Any, target_key: str) -> bool:
    if isinstance(value, dict):
        return any(
            key == target_key or _contains_key(item, target_key)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(item, target_key) for item in value)
    return False


def _matches_existing(existing: Any, update_payload: dict[str, Any]) -> bool:
    for field_name, expected in update_payload.items():
        if getattr(existing, field_name) != expected:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
