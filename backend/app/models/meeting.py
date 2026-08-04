from datetime import datetime
from typing import Any

from backend.app.models.base import Base
from backend.app.models.mixins import TimestampMixin
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Meeting(TimestampMixin, Base):
    __tablename__ = "meetings"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(255))
    participants: Mapped[list[str]] = mapped_column(JSON, default=list)
    basic_info: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    audio_filename: Mapped[str | None] = mapped_column(String(255))
    audio_storage_path: Mapped[str | None] = mapped_column(String(1024))
    audio_mime_type: Mapped[str | None] = mapped_column(String(128))
    audio_size: Mapped[int | None] = mapped_column(Integer)
    audio_hash: Mapped[str | None] = mapped_column(String(64))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    current_version: Mapped[int] = mapped_column(Integer, default=0)
    approved_version: Mapped[int | None] = mapped_column(Integer)

    segments: Mapped[list["TranscriptSegment"]] = relationship(
        cascade="all, delete-orphan", order_by="TranscriptSegment.start_time"
    )
    speaker_mappings: Mapped[list["SpeakerMapping"]] = relationship(cascade="all, delete-orphan")


class TranscriptSegment(TimestampMixin, Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    speaker_id: Mapped[str] = mapped_column(String(64))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    audit_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SpeakerMapping(TimestampMixin, Base):
    __tablename__ = "speaker_mappings"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    speaker_id: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(255))
    representative_segment_id: Mapped[int | None] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="SET NULL")
    )


class MeetingMinutes(TimestampMixin, Base):
    __tablename__ = "meeting_minutes"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(Text, default="")
    topics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    unresolved_questions: Mapped[list[str]] = mapped_column(JSON, default=list)
    risks: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="reviewing")
    reviewer_id: Mapped[str | None] = mapped_column(String(128))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    content_revision: Mapped[int] = mapped_column(Integer, default=1)


class MeetingDecision(TimestampMixin, Base):
    __tablename__ = "meeting_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    minutes_id: Mapped[int] = mapped_column(ForeignKey("meeting_minutes.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    decision_type: Mapped[str] = mapped_column(String(32), default="confirmed")
    source_segment_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    evidence_timestamps: Mapped[list[float]] = mapped_column(JSON, default=list)


class MeetingActionItem(TimestampMixin, Base):
    __tablename__ = "meeting_action_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    minutes_id: Mapped[int] = mapped_column(ForeignKey("meeting_minutes.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    owner: Mapped[str] = mapped_column(String(255), default="待确认")
    deadline: Mapped[str] = mapped_column(String(64), default="待确认")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    source_segment_ids: Mapped[list[int]] = mapped_column(JSON, default=list)


class EmailDraft(TimestampMixin, Base):
    __tablename__ = "email_drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    minutes_version: Mapped[int] = mapped_column(Integer)
    minutes_revision: Mapped[int] = mapped_column(Integer)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    to_addresses: Mapped[list[str]] = mapped_column(JSON, default=list)
    cc_addresses: Mapped[list[str]] = mapped_column(JSON, default=list)
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="draft")


class EmailDelivery(TimestampMixin, Base):
    __tablename__ = "email_deliveries"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    draft_id: Mapped[int] = mapped_column(ForeignKey("email_drafts.id"))
    provider: Mapped[str] = mapped_column(String(64))
    to_addresses: Mapped[list[str]] = mapped_column(JSON)
    cc_addresses: Mapped[list[str]] = mapped_column(JSON, default=list)
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    provider_message_id: Mapped[str | None] = mapped_column(String(255))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_message: Mapped[str | None] = mapped_column(Text)
    attempt: Mapped[int] = mapped_column(Integer, default=1)


class MeetingAuditEvent(Base):
    __tablename__ = "meeting_audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    actor_id: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(128))
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
