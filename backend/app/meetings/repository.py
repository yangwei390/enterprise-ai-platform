from typing import Any

from backend.app.models import (
    EmailDelivery,
    EmailDraft,
    Meeting,
    MeetingActionItem,
    MeetingAuditEvent,
    MeetingDecision,
    MeetingMinutes,
    SpeakerMapping,
    TranscriptSegment,
)
from sqlalchemy import delete, select
from sqlalchemy.orm import Session


class MeetingRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def commit(self) -> None:
        self.db.commit()

    def create(self, owner_id: str, data: dict[str, Any]) -> Meeting:
        meeting = Meeting(owner_id=owner_id, **data)
        self.db.add(meeting)
        self.db.commit()
        self.db.refresh(meeting)
        return meeting

    def get(self, meeting_id: int, owner_id: str) -> Meeting | None:
        return self.db.scalar(
            select(Meeting).where(Meeting.id == meeting_id, Meeting.owner_id == owner_id)
        )

    def list_by_owner(self, owner_id: str) -> list[Meeting]:
        return list(
            self.db.scalars(
                select(Meeting)
                .where(Meeting.owner_id == owner_id)
                .order_by(Meeting.created_at.desc())
            )
        )

    def latest_minutes(self, meeting_id: int) -> MeetingMinutes | None:
        return self.db.scalar(
            select(MeetingMinutes)
            .where(MeetingMinutes.meeting_id == meeting_id)
            .order_by(MeetingMinutes.version.desc())
            .limit(1)
        )

    def minutes_version(self, meeting_id: int, version: int) -> MeetingMinutes | None:
        return self.db.scalar(
            select(MeetingMinutes).where(
                MeetingMinutes.meeting_id == meeting_id, MeetingMinutes.version == version
            )
        )

    def decisions(self, minutes_id: int) -> list[MeetingDecision]:
        return list(
            self.db.scalars(select(MeetingDecision).where(MeetingDecision.minutes_id == minutes_id))
        )

    def actions(self, minutes_id: int) -> list[MeetingActionItem]:
        return list(
            self.db.scalars(
                select(MeetingActionItem).where(MeetingActionItem.minutes_id == minutes_id)
            )
        )

    def latest_draft(self, meeting_id: int) -> EmailDraft | None:
        return self.db.scalar(
            select(EmailDraft)
            .where(EmailDraft.meeting_id == meeting_id)
            .order_by(EmailDraft.id.desc())
            .limit(1)
        )

    def latest_delivery(self, meeting_id: int) -> EmailDelivery | None:
        return self.db.scalar(
            select(EmailDelivery)
            .where(EmailDelivery.meeting_id == meeting_id)
            .order_by(EmailDelivery.id.desc())
            .limit(1)
        )

    def add_audit(
        self,
        meeting_id: int,
        actor_id: str,
        action: str,
        entity_type: str,
        entity_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.db.add(
            MeetingAuditEvent(
                meeting_id=meeting_id,
                actor_id=actor_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                metadata_json=metadata or {},
            )
        )

    def replace_segments(self, meeting_id: int, segments: list[dict[str, Any]]) -> None:
        self.db.execute(delete(TranscriptSegment).where(TranscriptSegment.meeting_id == meeting_id))
        self.db.add_all([TranscriptSegment(meeting_id=meeting_id, **item) for item in segments])

    def replace_mappings(self, meeting_id: int, mappings: list[dict[str, Any]]) -> None:
        self.db.execute(delete(SpeakerMapping).where(SpeakerMapping.meeting_id == meeting_id))
        self.db.add_all([SpeakerMapping(meeting_id=meeting_id, **item) for item in mappings])

    def replace_minutes_children(
        self,
        minutes: MeetingMinutes,
        decisions: list[dict[str, Any]],
        actions: list[dict[str, Any]],
    ) -> None:
        self.db.execute(delete(MeetingDecision).where(MeetingDecision.minutes_id == minutes.id))
        self.db.execute(delete(MeetingActionItem).where(MeetingActionItem.minutes_id == minutes.id))
        self.db.add_all(
            [
                MeetingDecision(meeting_id=minutes.meeting_id, minutes_id=minutes.id, **x)
                for x in decisions
            ]
        )
        self.db.add_all(
            [
                MeetingActionItem(meeting_id=minutes.meeting_id, minutes_id=minutes.id, **x)
                for x in actions
            ]
        )

    def delete_meeting(self, meeting: Meeting) -> None:
        self.db.delete(meeting)
        self.db.commit()

    def add_delivery(self, data: dict[str, Any]) -> EmailDelivery:
        delivery = EmailDelivery(**data)
        self.db.add(delivery)
        self.db.flush()
        return delivery
