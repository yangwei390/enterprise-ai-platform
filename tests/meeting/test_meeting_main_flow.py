from pathlib import Path

import pytest
from backend.app.config.settings import settings
from backend.app.exceptions import BusinessException
from backend.app.meetings.providers import ASRSegment
from backend.app.meetings.repository import MeetingRepository
from backend.app.meetings.schemas import EmailDraftUpdate, MeetingCreate, MinutesUpdate
from backend.app.meetings.service import MeetingService
from backend.app.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class FakeASR:
    def transcribe(self, _: Path):
        return (
            [
                ASRSegment("Speaker 1", 0, 4, "我们决定周五发布。"),
                ASRSegment("Speaker 2", 4, 8, "我来准备发布清单。"),
            ],
            8.0,
        )


class FakeWorkflow:
    def run(self, segments):
        return {
            "summary": "团队确认周五发布。",
            "topics": [{"title": "发布"}],
            "decisions": [
                {
                    "content": "周五发布",
                    "decision_type": "confirmed",
                    "source_segment_ids": [segments[0].id],
                    "evidence_timestamps": [0.0],
                }
            ],
            "action_items": [
                {
                    "content": "准备发布清单",
                    "owner": "待确认",
                    "deadline": "待确认",
                    "status": "pending",
                    "source_segment_ids": [segments[1].id],
                }
            ],
            "unresolved_questions": [],
            "risks": [],
        }


class FakeEmail:
    name = "test-email"

    def __init__(self):
        self.sent = 0

    def send(self, to, cc, subject, body):
        self.sent += 1
        return "message-1"


class FailingASR:
    def transcribe(self, _: Path):
        raise BusinessException(52002, "provider unavailable")


@pytest.fixture
def service(tmp_path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    meeting_tables = [
        table
        for table in Base.metadata.sorted_tables
        if table.name
        in {
            "meetings",
            "transcript_segments",
            "speaker_mappings",
            "meeting_minutes",
            "meeting_decisions",
            "meeting_action_items",
            "email_drafts",
            "email_deliveries",
            "meeting_audit_events",
        }
    ]
    Base.metadata.create_all(engine, tables=meeting_tables)
    db = Session(engine)
    settings.UPLOAD_DIR = str(tmp_path)
    email = FakeEmail()
    result = MeetingService(
        MeetingRepository(db), asr=FakeASR(), email=email, workflow=FakeWorkflow()
    )
    result.test_email = email
    yield result
    db.close()


def prepare_audio(service: MeetingService, meeting_id: int) -> None:
    path = Path(settings.UPLOAD_DIR) / "audio.mp3"
    path.write_bytes(b"ID3-test")
    meeting = service.get("owner-a", meeting_id)
    meeting.audio_storage_path = "audio.mp3"
    meeting.audio_filename = "meeting.mp3"
    meeting.audio_mime_type = "audio/mpeg"
    service.repository.commit()


def test_complete_review_and_confirmed_send(service: MeetingService):
    meeting = service.create("owner-a", MeetingCreate(title="Release sync"))
    prepare_audio(service, meeting.id)
    processed = service.process("owner-a", meeting.id)
    assert processed.status == "draft_ready"
    transcript = service.transcript("owner-a", meeting.id)
    assert transcript[0]["speaker_id"] == "Speaker 1"

    minutes = service.minutes_payload("owner-a", meeting.id)
    assert minutes and minutes["decisions"][0]["source_segment_ids"] == [transcript[0]["id"]]
    service.approve("owner-a", meeting.id)
    draft = service.create_email_draft("owner-a", meeting.id)
    draft = service.update_email_draft(
        "owner-a",
        meeting.id,
        EmailDraftUpdate(to_addresses=["team@example.com"], subject=draft.subject, body=draft.body),
    )
    delivery = service.send_email("owner-a", meeting.id, True, draft.revision)
    assert delivery.status == "sent"
    assert service.get("owner-a", meeting.id).status == "sent"
    assert service.test_email.sent == 1


def test_owner_isolation_and_unapproved_send(service: MeetingService):
    meeting = service.create("owner-a", MeetingCreate(title="Private"))
    with pytest.raises(BusinessException, match="会议不存在"):
        service.get("owner-b", meeting.id)
    with pytest.raises(BusinessException, match="尚未批准"):
        service.create_email_draft("owner-a", meeting.id)


def test_edit_after_approval_creates_new_review_version(service: MeetingService):
    meeting = service.create("owner-a", MeetingCreate(title="Versioning"))
    prepare_audio(service, meeting.id)
    service.process("owner-a", meeting.id)
    service.approve("owner-a", meeting.id)
    current = service.minutes_payload("owner-a", meeting.id)
    assert current
    updated = service.update_minutes(
        "owner-a",
        meeting.id,
        MinutesUpdate(
            summary="edited",
            topics=current["topics"],
            decisions=current["decisions"],
            action_items=current["action_items"],
        ),
    )
    assert updated["version"] == 2
    assert service.get("owner-a", meeting.id).approved_version is None
    with pytest.raises(BusinessException, match="尚未批准"):
        service.create_email_draft("owner-a", meeting.id)


def test_send_rejects_missing_confirmation_and_stale_revision(service: MeetingService):
    meeting = service.create("owner-a", MeetingCreate(title="Safe send"))
    prepare_audio(service, meeting.id)
    service.process("owner-a", meeting.id)
    service.approve("owner-a", meeting.id)
    draft = service.create_email_draft("owner-a", meeting.id)
    draft = service.update_email_draft(
        "owner-a",
        meeting.id,
        EmailDraftUpdate(to_addresses=["a@example.com"], subject=draft.subject, body=draft.body),
    )
    with pytest.raises(BusinessException, match="明确确认"):
        service.send_email("owner-a", meeting.id, False, draft.revision)
    with pytest.raises(BusinessException, match="草稿已变化"):
        service.send_email("owner-a", meeting.id, True, draft.revision - 1)


def test_processing_failure_retry_and_duplicate_guard(service: MeetingService):
    meeting = service.create("owner-a", MeetingCreate(title="Retry"))
    prepare_audio(service, meeting.id)
    service._asr = FailingASR()
    with pytest.raises(BusinessException, match="provider unavailable"):
        service.process("owner-a", meeting.id)
    assert service.get("owner-a", meeting.id).status == "failed"
    service._asr = FakeASR()
    service.process("owner-a", meeting.id)
    with pytest.raises(BusinessException, match="不允许重复处理"):
        service.process("owner-a", meeting.id)
