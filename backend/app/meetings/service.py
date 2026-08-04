from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.meetings.providers import (
    ASRProvider,
    EmailProvider,
    get_asr_provider,
    get_email_provider,
)
from backend.app.meetings.repository import MeetingRepository
from backend.app.meetings.schemas import (
    EmailDraftUpdate,
    MeetingCreate,
    MinutesUpdate,
    SpeakerMappingInput,
)
from backend.app.meetings.workflow import MeetingMinutesWorkflow
from backend.app.models import EmailDraft, Meeting, MeetingMinutes
from backend.app.storage import LocalStorageService
from fastapi import UploadFile

ACTIVE_STATUSES = {"transcribing", "generating_minutes", "sending"}
DELETABLE_STATUSES = {"uploaded", "failed"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class MeetingService:
    def __init__(
        self,
        repository: MeetingRepository,
        asr: ASRProvider | None = None,
        email: EmailProvider | None = None,
        workflow: MeetingMinutesWorkflow | None = None,
    ) -> None:
        self.repository = repository
        self._asr = asr
        self._email = email
        self.workflow = workflow or MeetingMinutesWorkflow()

    def create(self, owner_id: str, data: MeetingCreate) -> Meeting:
        meeting = self.repository.create(owner_id, {**data.model_dump(), "status": "uploaded"})
        self.repository.add_audit(meeting.id, owner_id, "meeting.created", "meeting", meeting.id)
        self.repository.commit()
        return meeting

    def get(self, owner_id: str, meeting_id: int) -> Meeting:
        meeting = self.repository.get(meeting_id, owner_id)
        if meeting is None:
            raise BusinessException(40420, "会议不存在")
        return meeting

    def upload_audio(self, owner_id: str, meeting_id: int, file: UploadFile) -> Meeting:
        meeting = self.get(owner_id, meeting_id)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp3", ".wav", ".m4a"} or file.content_type not in ALLOWED_AUDIO_TYPES:
            raise BusinessException(40020, "仅支持 MP3/WAV/M4A 音频")
        if meeting.status in ACTIVE_STATUSES:
            raise BusinessException(40920, "会议正在处理中，不能替换音频")
        stored = LocalStorageService().save_upload_file(file)
        meeting.audio_filename = stored["original_filename"]
        meeting.audio_storage_path = stored["storage_path"]
        meeting.audio_mime_type = stored["mime_type"]
        meeting.audio_size = stored["file_size"]
        meeting.audio_hash = stored["file_hash"]
        meeting.status, meeting.progress, meeting.error_message = "uploaded", 5, None
        self.repository.add_audit(
            meeting.id,
            owner_id,
            "audio.uploaded",
            "meeting",
            meeting.id,
            {"size": meeting.audio_size, "sha256": meeting.audio_hash},
        )
        self.repository.commit()
        return meeting

    def audio_path(self, owner_id: str, meeting_id: int) -> Path:
        meeting = self.get(owner_id, meeting_id)
        if not meeting.audio_storage_path:
            raise BusinessException(40421, "会议音频不存在")
        base = Path(settings.UPLOAD_DIR)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        path = (base / meeting.audio_storage_path).resolve()
        if base.resolve() not in path.parents or not path.exists():
            raise BusinessException(40421, "会议音频不存在")
        return path

    def process(self, owner_id: str, meeting_id: int) -> Meeting:
        meeting = self.get(owner_id, meeting_id)
        if meeting.status in ACTIVE_STATUSES:
            raise BusinessException(40921, "该会议已在处理中")
        if meeting.status not in {"uploaded", "failed"}:
            raise BusinessException(40926, "当前状态不允许重复处理")
        if not meeting.audio_storage_path:
            raise BusinessException(40021, "请先上传音频")
        try:
            meeting.status, meeting.progress, meeting.error_message = "transcribing", 15, None
            self.repository.commit()
            asr = self._asr or get_asr_provider()
            segments, duration = asr.transcribe(self.audio_path(owner_id, meeting_id))
            self.repository.replace_segments(
                meeting.id,
                [
                    {
                        "speaker_id": s.speaker_id,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "text": s.text,
                        "confidence": s.confidence,
                        "audit_payload": s.audit_payload or {},
                    }
                    for s in segments
                ],
            )
            meeting.duration_seconds = duration or max(segment.end_time for segment in segments)
            meeting.status, meeting.progress = "generating_minutes", 60
            self.repository.commit()
            self.repository.db.refresh(meeting)
            result = self.workflow.run(meeting.segments)
            version = meeting.current_version + 1
            minutes = MeetingMinutes(
                meeting_id=meeting.id,
                version=version,
                summary=str(result.get("summary", "")),
                topics=result.get("topics", []),
                unresolved_questions=result.get("unresolved_questions", []),
                risks=result.get("risks", []),
                status="reviewing",
            )
            self.repository.db.add(minutes)
            self.repository.db.flush()
            self.repository.replace_minutes_children(
                minutes, result.get("decisions", []), result.get("action_items", [])
            )
            meeting.current_version, meeting.status, meeting.progress = version, "draft_ready", 100
            self.repository.add_audit(
                meeting.id,
                owner_id,
                "minutes.generated",
                "minutes",
                minutes.id,
                {"version": version},
            )
            self.repository.commit()
        except Exception as exc:
            meeting.status, meeting.error_message = "failed", self._safe_error(exc)
            self.repository.commit()
            raise
        logger.info("Meeting processing succeeded | meeting_id=%s", meeting.id)
        return meeting

    def transcript(self, owner_id: str, meeting_id: int) -> list[dict[str, Any]]:
        meeting = self.get(owner_id, meeting_id)
        names = {m.speaker_id: m.display_name for m in meeting.speaker_mappings}
        return [
            {
                "id": s.id,
                "speaker_id": s.speaker_id,
                "speaker_name": names.get(s.speaker_id, s.speaker_id),
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "confidence": s.confidence,
            }
            for s in meeting.segments
        ]

    def minutes_payload(self, owner_id: str, meeting_id: int) -> dict[str, Any] | None:
        meeting = self.get(owner_id, meeting_id)
        minutes = self.repository.latest_minutes(meeting.id)
        if minutes is None:
            return None
        return {
            "id": minutes.id,
            "version": minutes.version,
            "content_revision": minutes.content_revision,
            "status": minutes.status,
            "summary": minutes.summary,
            "topics": minutes.topics,
            "unresolved_questions": minutes.unresolved_questions,
            "risks": minutes.risks,
            "reviewer_id": minutes.reviewer_id,
            "approved_at": minutes.approved_at,
            "decisions": [self._columns(x) for x in self.repository.decisions(minutes.id)],
            "action_items": [self._columns(x) for x in self.repository.actions(minutes.id)],
        }

    def update_speakers(
        self, owner_id: str, meeting_id: int, mappings: list[SpeakerMappingInput]
    ) -> None:
        meeting = self.get(owner_id, meeting_id)
        valid_speakers = {s.speaker_id for s in meeting.segments}
        if any(item.speaker_id not in valid_speakers for item in mappings):
            raise BusinessException(40022, "说话人映射包含未知 Speaker")
        replacements = {item.speaker_id: item.display_name for item in mappings}
        minutes = self.repository.latest_minutes(meeting.id)
        if minutes is not None:
            minutes.summary = self._replace_names(minutes.summary, replacements)
            minutes.topics = self._replace_names(minutes.topics, replacements)
            for decision in self.repository.decisions(minutes.id):
                decision.content = self._replace_names(decision.content, replacements)
            for action in self.repository.actions(minutes.id):
                action.content = self._replace_names(action.content, replacements)
                action.owner = self._replace_names(action.owner, replacements)
            minutes.content_revision += 1
        draft = self.repository.latest_draft(meeting.id)
        if draft is not None and draft.status == "draft":
            draft.body = self._replace_names(draft.body, replacements)
            draft.revision += 1
        self.repository.replace_mappings(meeting.id, [item.model_dump() for item in mappings])
        self._invalidate_approval(meeting)
        self.repository.add_audit(meeting.id, owner_id, "speakers.updated", "meeting", meeting.id)
        self.repository.commit()

    def update_minutes(self, owner_id: str, meeting_id: int, data: MinutesUpdate) -> dict[str, Any]:
        meeting = self.get(owner_id, meeting_id)
        current = self.repository.latest_minutes(meeting.id)
        if current is None:
            raise BusinessException(40422, "会议纪要不存在")
        if current.status == "approved":
            current = MeetingMinutes(
                meeting_id=meeting.id,
                version=current.version + 1,
                summary=data.summary,
                topics=data.topics,
                unresolved_questions=data.unresolved_questions,
                risks=data.risks,
                status="reviewing",
            )
            self.repository.db.add(current)
            self.repository.db.flush()
            meeting.current_version = current.version
        else:
            current.summary, current.topics = data.summary, data.topics
            current.unresolved_questions, current.risks = data.unresolved_questions, data.risks
            current.content_revision += 1
        self.repository.replace_minutes_children(
            current,
            [x.model_dump() for x in data.decisions],
            [x.model_dump() for x in data.action_items],
        )
        meeting.status, meeting.approved_version = "reviewing", None
        self.repository.add_audit(
            meeting.id,
            owner_id,
            "minutes.updated",
            "minutes",
            current.id,
            {"version": current.version, "revision": current.content_revision},
        )
        self.repository.commit()
        return self.minutes_payload(owner_id, meeting_id) or {}

    def approve(self, owner_id: str, meeting_id: int) -> dict[str, Any]:
        meeting = self.get(owner_id, meeting_id)
        minutes = self.repository.latest_minutes(meeting.id)
        if minutes is None:
            raise BusinessException(40422, "会议纪要不存在")
        minutes.status, minutes.reviewer_id, minutes.approved_at = (
            "approved",
            owner_id,
            datetime.utcnow(),
        )
        meeting.status, meeting.approved_version = "approved", minutes.version
        self.repository.add_audit(
            meeting.id,
            owner_id,
            "minutes.approved",
            "minutes",
            minutes.id,
            {"version": minutes.version, "revision": minutes.content_revision},
        )
        self.repository.commit()
        return self.minutes_payload(owner_id, meeting_id) or {}

    def create_email_draft(self, owner_id: str, meeting_id: int) -> EmailDraft:
        meeting, minutes = self._approved(owner_id, meeting_id)
        payload = self.minutes_payload(owner_id, meeting_id) or {}
        body = self._render_email(meeting, payload)
        draft = EmailDraft(
            meeting_id=meeting.id,
            minutes_version=minutes.version,
            minutes_revision=minutes.content_revision,
            to_addresses=[],
            cc_addresses=[],
            subject=f"会议纪要：{meeting.title}",
            body=body,
        )
        self.repository.db.add(draft)
        self.repository.db.flush()
        meeting.status = "email_draft_ready"
        self.repository.add_audit(
            meeting.id, owner_id, "email_draft.created", "email_draft", draft.id
        )
        self.repository.commit()
        self.repository.db.refresh(draft)
        return draft

    def update_email_draft(
        self, owner_id: str, meeting_id: int, data: EmailDraftUpdate
    ) -> EmailDraft:
        self.get(owner_id, meeting_id)
        draft = self.repository.latest_draft(meeting_id)
        if draft is None or draft.status != "draft":
            raise BusinessException(40423, "可编辑邮件草稿不存在")
        for key, value in data.model_dump().items():
            setattr(draft, key, value)
        draft.revision += 1
        self.repository.add_audit(
            meeting_id,
            owner_id,
            "email_draft.updated",
            "email_draft",
            draft.id,
            {"revision": draft.revision},
        )
        self.repository.commit()
        self.repository.db.refresh(draft)
        return draft

    def send_email(
        self, owner_id: str, meeting_id: int, confirm: bool, expected_revision: int
    ) -> Any:
        if not confirm:
            raise BusinessException(40023, "发送邮件需要明确确认")
        meeting, minutes = self._approved(owner_id, meeting_id)
        draft = self.repository.latest_draft(meeting_id)
        if (
            draft is None
            or not draft.to_addresses
            or not draft.subject.strip()
            or not draft.body.strip()
        ):
            raise BusinessException(40024, "邮件草稿收件人、主题和正文不能为空")
        if draft.revision != expected_revision:
            raise BusinessException(40922, "邮件草稿已变化，请重新确认")
        if (
            draft.minutes_version != minutes.version
            or draft.minutes_revision != minutes.content_revision
        ):
            raise BusinessException(40923, "批准后内容已变化，必须重新审核并生成草稿")
        if any(not EMAIL_RE.match(address) for address in draft.to_addresses + draft.cc_addresses):
            raise BusinessException(40025, "收件人邮箱格式无效")
        meeting.status = "sending"
        self.repository.commit()
        provider = self._email or get_email_provider()
        previous = self.repository.latest_delivery(meeting.id)
        delivery = self.repository.add_delivery(
            {
                "meeting_id": meeting.id,
                "draft_id": draft.id,
                "provider": provider.name,
                "to_addresses": draft.to_addresses,
                "cc_addresses": draft.cc_addresses,
                "subject": draft.subject,
                "body": draft.body,
                "status": "sending",
                "attempt": (previous.attempt + 1) if previous else 1,
            }
        )
        try:
            delivery.provider_message_id = provider.send(
                draft.to_addresses, draft.cc_addresses, draft.subject, draft.body
            )
            delivery.status, delivery.sent_at, draft.status, meeting.status = (
                "sent",
                datetime.utcnow(),
                "sent",
                "sent",
            )
        except Exception as exc:
            delivery.status, delivery.error_message, meeting.status = (
                "failed",
                self._safe_error(exc),
                "failed",
            )
            meeting.error_message = delivery.error_message
            self.repository.commit()
            raise
        self.repository.add_audit(meeting.id, owner_id, "email.sent", "email_delivery", delivery.id)
        self.repository.commit()
        return delivery

    def delete(self, owner_id: str, meeting_id: int) -> None:
        meeting = self.get(owner_id, meeting_id)
        if meeting.status not in DELETABLE_STATUSES:
            raise BusinessException(40924, "仅可删除未处理或失败会议")
        self.repository.delete_meeting(meeting)

    def _approved(self, owner_id: str, meeting_id: int) -> tuple[Meeting, MeetingMinutes]:
        meeting = self.get(owner_id, meeting_id)
        if meeting.approved_version is None or meeting.approved_version != meeting.current_version:
            raise BusinessException(40925, "当前会议纪要版本尚未批准")
        minutes = self.repository.minutes_version(meeting.id, meeting.approved_version)
        if minutes is None or minutes.status != "approved":
            raise BusinessException(40925, "当前会议纪要版本尚未批准")
        return meeting, minutes

    def _invalidate_approval(self, meeting: Meeting) -> None:
        if meeting.approved_version is not None:
            meeting.approved_version, meeting.status = None, "reviewing"

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        return (
            exc.message if isinstance(exc, BusinessException) else f"处理失败：{type(exc).__name__}"
        )

    @staticmethod
    def _columns(obj: Any) -> dict[str, Any]:
        return {
            column.name: getattr(obj, column.name)
            for column in obj.__table__.columns
            if column.name not in {"created_at", "updated_at", "meeting_id", "minutes_id"}
        }

    @staticmethod
    def _render_email(meeting: Meeting, payload: dict[str, Any]) -> str:
        lines = [f"会议：{meeting.title}", "", "摘要", payload.get("summary", ""), "", "已确认决策"]
        lines.extend(f"- {x['content']}" for x in payload.get("decisions", []))
        lines.append("\n行动项")
        lines.extend(
            f"- {x['content']}（负责人：{x['owner']}；截止：{x['deadline']}）"
            for x in payload.get("action_items", [])
        )
        return "\n".join(lines)

    @staticmethod
    def _replace_names(value: Any, replacements: dict[str, str]) -> Any:
        if isinstance(value, str):
            for source, target in replacements.items():
                value = value.replace(source, target)
            return value
        if isinstance(value, list):
            return [MeetingService._replace_names(item, replacements) for item in value]
        if isinstance(value, dict):
            return {
                key: MeetingService._replace_names(item, replacements)
                for key, item in value.items()
            }
        return value
