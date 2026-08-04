import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from backend.app.db.session import SessionLocal, get_db
from backend.app.exceptions import BusinessException
from backend.app.meetings.repository import MeetingRepository
from backend.app.meetings.schemas import (
    EmailDraftUpdate,
    MeetingCreate,
    MeetingResponse,
    MinutesUpdate,
    SendEmailRequest,
    SpeakerMappingInput,
)
from backend.app.meetings.service import MeetingService
from backend.app.schemas import ApiResponse, success
from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter(prefix="/meetings", tags=["meeting-minutes"])
OwnerId = Annotated[str, Header(alias="X-Owner-Id", min_length=1, max_length=128)]


def get_service(db: Session = Depends(get_db)) -> MeetingService:
    return MeetingService(MeetingRepository(db))


def _run_process(owner_id: str, meeting_id: int) -> None:
    db = SessionLocal()
    try:
        MeetingService(MeetingRepository(db)).process(owner_id, meeting_id)
    finally:
        db.close()


def _model_dict(value: Any) -> dict[str, Any]:
    return {column.name: getattr(value, column.name) for column in value.__table__.columns}


@router.post("", response_model=ApiResponse)
def create_meeting(
    data: MeetingCreate, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    return success(data=MeetingResponse.model_validate(service.create(owner_id, data)))


@router.get("", response_model=ApiResponse)
def list_meetings(owner_id: OwnerId, service: MeetingService = Depends(get_service)) -> ApiResponse:
    items = [MeetingResponse.model_validate(x) for x in service.repository.list_by_owner(owner_id)]
    return success(data={"items": items, "total": len(items)})


@router.get("/{meeting_id}", response_model=ApiResponse)
def get_meeting(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    return success(data=MeetingResponse.model_validate(service.get(owner_id, meeting_id)))


@router.post("/{meeting_id}/audio", response_model=ApiResponse)
def upload_audio(
    meeting_id: int,
    owner_id: OwnerId,
    file: UploadFile = File(...),
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    return success(
        data=MeetingResponse.model_validate(service.upload_audio(owner_id, meeting_id, file))
    )


@router.get("/{meeting_id}/audio")
def stream_audio(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> FileResponse:
    meeting = service.get(owner_id, meeting_id)
    return FileResponse(
        service.audio_path(owner_id, meeting_id),
        media_type=meeting.audio_mime_type,
        filename=meeting.audio_filename,
    )


@router.post("/{meeting_id}/process", response_model=ApiResponse)
def process_meeting(
    meeting_id: int,
    owner_id: OwnerId,
    tasks: BackgroundTasks,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    meeting = service.get(owner_id, meeting_id)
    if meeting.status in {"transcribing", "generating_minutes", "sending"}:
        return success(data=MeetingResponse.model_validate(meeting), message="already processing")
    if meeting.status not in {"uploaded", "failed"}:
        raise BusinessException(40926, "当前状态不允许重复处理")
    tasks.add_task(_run_process, owner_id, meeting_id)
    return success(data={"meeting_id": meeting_id, "accepted": True})


@router.post("/{meeting_id}/retry", response_model=ApiResponse)
def retry_meeting(
    meeting_id: int,
    owner_id: OwnerId,
    tasks: BackgroundTasks,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    meeting = service.get(owner_id, meeting_id)
    if meeting.status != "failed":
        return success(data=MeetingResponse.model_validate(meeting), message="retry not required")
    tasks.add_task(_run_process, owner_id, meeting_id)
    return success(data={"meeting_id": meeting_id, "accepted": True})


@router.get("/{meeting_id}/transcript", response_model=ApiResponse)
def get_transcript(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    return success(data={"segments": service.transcript(owner_id, meeting_id)})


@router.get("/{meeting_id}/minutes", response_model=ApiResponse)
def get_minutes(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    return success(data=service.minutes_payload(owner_id, meeting_id))


@router.put("/{meeting_id}/speakers", response_model=ApiResponse)
def update_speakers(
    meeting_id: int,
    data: list[SpeakerMappingInput],
    owner_id: OwnerId,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    service.update_speakers(owner_id, meeting_id, data)
    return success(data={"segments": service.transcript(owner_id, meeting_id)})


@router.put("/{meeting_id}/minutes", response_model=ApiResponse)
def update_minutes(
    meeting_id: int,
    data: MinutesUpdate,
    owner_id: OwnerId,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    return success(data=service.update_minutes(owner_id, meeting_id, data))


@router.post("/{meeting_id}/approve", response_model=ApiResponse)
def approve_minutes(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    return success(data=service.approve(owner_id, meeting_id))


@router.post("/{meeting_id}/email-draft", response_model=ApiResponse)
def create_email_draft(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    return success(data=_model_dict(service.create_email_draft(owner_id, meeting_id)))


@router.put("/{meeting_id}/email-draft", response_model=ApiResponse)
def update_email_draft(
    meeting_id: int,
    data: EmailDraftUpdate,
    owner_id: OwnerId,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    return success(data=_model_dict(service.update_email_draft(owner_id, meeting_id, data)))


@router.post("/{meeting_id}/send", response_model=ApiResponse)
def send_email(
    meeting_id: int,
    data: SendEmailRequest,
    owner_id: OwnerId,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    delivery = service.send_email(owner_id, meeting_id, data.confirm, data.expected_draft_revision)
    return success(data=_model_dict(delivery))


@router.post("/{meeting_id}/email/retry", response_model=ApiResponse)
def retry_email(
    meeting_id: int,
    data: SendEmailRequest,
    owner_id: OwnerId,
    service: MeetingService = Depends(get_service),
) -> ApiResponse:
    delivery = service.send_email(owner_id, meeting_id, data.confirm, data.expected_draft_revision)
    return success(data=_model_dict(delivery))


@router.delete("/{meeting_id}", response_model=ApiResponse)
def delete_meeting(
    meeting_id: int, owner_id: OwnerId, service: MeetingService = Depends(get_service)
) -> ApiResponse:
    service.delete(owner_id, meeting_id)
    return success(data={"deleted": True})


@router.get("/{meeting_id}/events")
async def meeting_events(meeting_id: int, owner_id: OwnerId) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        last = None
        for _ in range(300):
            db = SessionLocal()
            try:
                meeting = MeetingService(MeetingRepository(db)).get(owner_id, meeting_id)
                current = (meeting.status, meeting.progress, meeting.error_message)
                if current != last:
                    payload = {"status": current[0], "progress": current[1], "error": current[2]}
                    yield f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    last = current
                if meeting.status in {
                    "draft_ready",
                    "approved",
                    "email_draft_ready",
                    "sent",
                    "failed",
                }:
                    break
            finally:
                db.close()
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream")
