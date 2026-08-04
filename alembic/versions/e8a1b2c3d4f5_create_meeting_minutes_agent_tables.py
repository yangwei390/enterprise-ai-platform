"""create meeting minutes agent tables

Revision ID: e8a1b2c3d4f5
Revises: 6d4e9a2f1b8c
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e8a1b2c3d4f5"
down_revision: str | Sequence[str] | None = "6d4e9a2f1b8c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "meetings", sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.String(128), nullable=False), sa.Column("title", sa.String(255), nullable=False),
        sa.Column("participants", sa.JSON(), nullable=False), sa.Column("basic_info", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text()), sa.Column("audio_filename", sa.String(255)),
        sa.Column("audio_storage_path", sa.String(1024)), sa.Column("audio_mime_type", sa.String(128)),
        sa.Column("audio_size", sa.Integer()), sa.Column("audio_hash", sa.String(64)),
        sa.Column("duration_seconds", sa.Float()), sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("approved_version", sa.Integer()), sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_meetings_owner_id", "meetings", ["owner_id"])
    op.create_index("ix_meetings_status", "meetings", ["status"])
    _create_child_tables()


def _timestamps() -> list[sa.Column]:
    return [sa.Column("created_at", sa.DateTime(), nullable=False), sa.Column("updated_at", sa.DateTime(), nullable=False)]


def _meeting_fk() -> sa.Column:
    return sa.Column("meeting_id", sa.Integer(), sa.ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)


def _create_child_tables() -> None:
    op.create_table("transcript_segments", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("speaker_id", sa.String(64), nullable=False), sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False), sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float()), sa.Column("audit_payload", sa.JSON(), nullable=False), *_timestamps())
    op.create_index("ix_transcript_segments_meeting_id", "transcript_segments", ["meeting_id"])
    op.create_table("speaker_mappings", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("speaker_id", sa.String(64), nullable=False), sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("representative_segment_id", sa.Integer(), sa.ForeignKey("transcript_segments.id", ondelete="SET NULL")), *_timestamps())
    op.create_index("ix_speaker_mappings_meeting_id", "speaker_mappings", ["meeting_id"])
    op.create_table("meeting_minutes", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("version", sa.Integer(), nullable=False), sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("topics", sa.JSON(), nullable=False), sa.Column("unresolved_questions", sa.JSON(), nullable=False),
        sa.Column("risks", sa.JSON(), nullable=False), sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reviewer_id", sa.String(128)), sa.Column("approved_at", sa.DateTime()),
        sa.Column("content_revision", sa.Integer(), nullable=False), *_timestamps())
    op.create_index("ix_meeting_minutes_meeting_id", "meeting_minutes", ["meeting_id"])
    op.create_table("meeting_decisions", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("minutes_id", sa.Integer(), sa.ForeignKey("meeting_minutes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False), sa.Column("decision_type", sa.String(32), nullable=False),
        sa.Column("source_segment_ids", sa.JSON(), nullable=False), sa.Column("evidence_timestamps", sa.JSON(), nullable=False), *_timestamps())
    op.create_index("ix_meeting_decisions_meeting_id", "meeting_decisions", ["meeting_id"])
    op.create_table("meeting_action_items", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("minutes_id", sa.Integer(), sa.ForeignKey("meeting_minutes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False), sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("deadline", sa.String(64), nullable=False), sa.Column("status", sa.String(32), nullable=False),
        sa.Column("source_segment_ids", sa.JSON(), nullable=False), *_timestamps())
    op.create_index("ix_meeting_action_items_meeting_id", "meeting_action_items", ["meeting_id"])
    op.create_table("email_drafts", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("minutes_version", sa.Integer(), nullable=False), sa.Column("minutes_revision", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False), sa.Column("to_addresses", sa.JSON(), nullable=False),
        sa.Column("cc_addresses", sa.JSON(), nullable=False), sa.Column("subject", sa.String(500), nullable=False),
        sa.Column("body", sa.Text(), nullable=False), sa.Column("status", sa.String(32), nullable=False), *_timestamps())
    op.create_index("ix_email_drafts_meeting_id", "email_drafts", ["meeting_id"])
    op.create_table("email_deliveries", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("draft_id", sa.Integer(), sa.ForeignKey("email_drafts.id"), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False), sa.Column("to_addresses", sa.JSON(), nullable=False),
        sa.Column("cc_addresses", sa.JSON(), nullable=False), sa.Column("subject", sa.String(500), nullable=False),
        sa.Column("body", sa.Text(), nullable=False), sa.Column("status", sa.String(32), nullable=False),
        sa.Column("provider_message_id", sa.String(255)), sa.Column("sent_at", sa.DateTime()),
        sa.Column("error_message", sa.Text()), sa.Column("attempt", sa.Integer(), nullable=False), *_timestamps())
    op.create_index("ix_email_deliveries_meeting_id", "email_deliveries", ["meeting_id"])
    op.create_table("meeting_audit_events", sa.Column("id", sa.Integer(), primary_key=True), _meeting_fk(),
        sa.Column("actor_id", sa.String(128), nullable=False), sa.Column("action", sa.String(128), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False), sa.Column("entity_id", sa.Integer()),
        sa.Column("metadata_json", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_index("ix_meeting_audit_events_meeting_id", "meeting_audit_events", ["meeting_id"])


def downgrade() -> None:
    for table in ["meeting_audit_events", "email_deliveries", "email_drafts", "meeting_action_items", "meeting_decisions", "meeting_minutes", "speaker_mappings", "transcript_segments"]:
        op.drop_table(table)
    op.drop_table("meetings")
