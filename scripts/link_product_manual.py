from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config.settings import settings  # noqa: E402
from backend.app.db.session import SessionLocal  # noqa: E402
from backend.app.repositories.product import ProductRepository  # noqa: E402
from backend.app.schemas.product import ProductDocumentLinkCreate  # noqa: E402
from backend.app.services.product import ProductService  # noqa: E402
from pydantic import ValidationError  # noqa: E402


class ProductManualLinkService(Protocol):
    def link_document(
        self,
        data: ProductDocumentLinkCreate,
        *,
        allowed_knowledge_base_ids: set[int],
    ) -> Any: ...


@dataclass(slots=True)
class ManualLinkResult:
    link_id: int | None
    product_code: str
    document_id: int
    dry_run: bool
    message: str


def run_link_product_manual(
    data: ProductDocumentLinkCreate,
    service: ProductManualLinkService,
    *,
    allowed_knowledge_base_ids: set[int],
    dry_run: bool = True,
) -> ManualLinkResult:
    if dry_run:
        return ManualLinkResult(
            link_id=None,
            product_code=data.product_code,
            document_id=data.document_id,
            dry_run=True,
            message="dry-run only; no database write performed",
        )

    link = service.link_document(
        data,
        allowed_knowledge_base_ids=allowed_knowledge_base_ids,
    )
    return ManualLinkResult(
        link_id=link.id,
        product_code=data.product_code,
        document_id=data.document_id,
        dry_run=False,
        message="linked",
    )


def build_service() -> ProductService:
    db = SessionLocal()
    return ProductService(ProductRepository(db))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bind a product to an uploaded manual document")
    parser.add_argument("--product-code", required=True)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument(
        "--document-type",
        default="manual",
        choices=["manual", "warranty", "policy", "other"],
    )
    parser.add_argument("--is-primary", action="store_true")
    parser.add_argument("--manual-version")
    parser.add_argument("--source-url")
    parser.add_argument(
        "--allowed-knowledge-base-id",
        type=int,
        action="append",
        required=True,
        dest="allowed_knowledge_base_ids",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="默认 dry-run；传入该参数才写入数据库",
    )
    args = parser.parse_args()

    print(_database_summary())
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    try:
        payload = ProductDocumentLinkCreate(
            product_code=args.product_code,
            document_id=args.document_id,
            document_type=args.document_type,
            is_primary=args.is_primary,
            manual_version=args.manual_version,
            source_url=args.source_url,
        )
        result = run_link_product_manual(
            payload,
            build_service(),
            allowed_knowledge_base_ids=set(args.allowed_knowledge_base_ids),
            dry_run=not args.apply,
        )
    except ValidationError as exc:
        print(f"manual link validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"manual link failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"product_code={result.product_code} "
        f"document_id={result.document_id} "
        f"link_id={result.link_id} "
        f"dry_run={result.dry_run} message={result.message}"
    )
    return 0


def _database_summary() -> str:
    return (
        "target_db="
        f"{settings.POSTGRES_USER}@{settings.POSTGRES_HOST}:"
        f"{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
