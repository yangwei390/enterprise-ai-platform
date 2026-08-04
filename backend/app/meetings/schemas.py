from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MeetingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    participants: list[str] = Field(default_factory=list)
    basic_info: dict[str, Any] = Field(default_factory=dict)


class SpeakerMappingInput(BaseModel):
    speaker_id: str
    display_name: str = Field(min_length=1, max_length=255)
    representative_segment_id: int | None = None


class DecisionInput(BaseModel):
    content: str
    decision_type: str = "confirmed"
    source_segment_ids: list[int] = Field(default_factory=list)
    evidence_timestamps: list[float] = Field(default_factory=list)


class ActionItemInput(BaseModel):
    content: str
    owner: str = "待确认"
    deadline: str = "待确认"
    status: str = "pending"
    source_segment_ids: list[int] = Field(default_factory=list)


class MinutesUpdate(BaseModel):
    summary: str
    topics: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    decisions: list[DecisionInput] = Field(default_factory=list)
    action_items: list[ActionItemInput] = Field(default_factory=list)


class EmailDraftUpdate(BaseModel):
    to_addresses: list[str]
    cc_addresses: list[str] = Field(default_factory=list)
    subject: str
    body: str

    @field_validator("to_addresses", "cc_addresses")
    @classmethod
    def validate_addresses(cls, values: list[str]) -> list[str]:
        if any(
            "@" not in value or value.startswith("@") or value.endswith("@") for value in values
        ):
            raise ValueError("收件人邮箱格式无效")
        return values


class SendEmailRequest(BaseModel):
    confirm: bool
    expected_draft_revision: int


class MeetingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    owner_id: str
    title: str
    participants: list[str]
    basic_info: dict[str, Any]
    status: str
    progress: int
    error_message: str | None
    audio_filename: str | None
    duration_seconds: float | None
    current_version: int
    approved_version: int | None
    created_at: datetime
    updated_at: datetime
