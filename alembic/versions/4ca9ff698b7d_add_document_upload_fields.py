"""add document upload fields

Revision ID: 4ca9ff698b7d
Revises: 13ed2fe01303
Create Date: 2026-07-04 17:27:05.952091

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '4ca9ff698b7d'
down_revision: Union[str, Sequence[str], None] = '13ed2fe01303'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("original_filename", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("mime_type", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("file_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "parse_status",
            sa.String(length=64),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "documents",
        sa.Column("parse_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "parse_message")
    op.drop_column("documents", "parse_status")
    op.drop_column("documents", "file_hash")
    op.drop_column("documents", "mime_type")
    op.drop_column("documents", "storage_path")
    op.drop_column("documents", "original_filename")
