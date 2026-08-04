from __future__ import annotations

import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from backend.app.config.settings import settings
from backend.app.exceptions import BusinessException
from openai import OpenAI  # type: ignore[reportMissingImports]


@dataclass
class ASRSegment:
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float | None = None
    audit_payload: dict[str, Any] | None = None


class ASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> tuple[list[ASRSegment], float | None]: ...


class OpenAICompatibleASRProvider(ASRProvider):
    def __init__(self) -> None:
        if not settings.MEETING_ASR_API_KEY:
            raise BusinessException(52001, "未配置 MEETING_ASR_API_KEY，无法执行真实转写")
        self.client = OpenAI(
            api_key=settings.MEETING_ASR_API_KEY,
            base_url=settings.MEETING_ASR_BASE_URL,
            timeout=settings.MEETING_ASR_TIMEOUT,
        )

    def transcribe(self, audio_path: Path) -> tuple[list[ASRSegment], float | None]:
        try:
            with audio_path.open("rb") as audio:
                response = self.client.audio.transcriptions.create(
                    model=settings.MEETING_ASR_MODEL,
                    file=audio,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
        except Exception as exc:
            raise BusinessException(52002, f"音频转写失败：{type(exc).__name__}") from exc
        payload = response.model_dump() if hasattr(response, "model_dump") else response
        raw_segments = payload.get("segments", []) if isinstance(payload, dict) else []
        segments = []
        for index, item in enumerate(raw_segments):
            speaker = item.get("speaker") or item.get("speaker_id")
            if settings.MEETING_ASR_REQUIRE_DIARIZATION and not speaker:
                raise BusinessException(52003, "ASR 返回结果不含说话人分离标签")
            speaker_id = str(speaker or "Speaker 1")
            if not speaker_id.lower().startswith("speaker"):
                speaker_id = f"Speaker {speaker_id}"
            segments.append(
                ASRSegment(
                    speaker_id=speaker_id,
                    start_time=float(item.get("start", 0)),
                    end_time=float(item.get("end", 0)),
                    text=str(item.get("text", "")).strip(),
                    confidence=item.get("confidence"),
                    audit_payload={"provider_segment_index": index},
                )
            )
        if not segments:
            raise BusinessException(52004, "ASR 未返回可用转写分段")
        duration = payload.get("duration") if isinstance(payload, dict) else None
        return segments, float(duration) if duration is not None else None


class EmailProvider(ABC):
    name: str

    @abstractmethod
    def send(self, to: list[str], cc: list[str], subject: str, body: str) -> str | None: ...


class SMTPEmailProvider(EmailProvider):
    name = "smtp"

    def __init__(self) -> None:
        if not settings.SMTP_HOST or not settings.SMTP_FROM_ADDRESS:
            raise BusinessException(53001, "未配置 SMTP_HOST/SMTP_FROM_ADDRESS，无法真实发送邮件")
        self.host = settings.SMTP_HOST

    def send(self, to: list[str], cc: list[str], subject: str, body: str) -> str | None:
        message = EmailMessage()
        message["From"] = settings.SMTP_FROM_ADDRESS
        message["To"] = ", ".join(to)
        if cc:
            message["Cc"] = ", ".join(cc)
        message["Subject"] = subject
        message.set_content(body)
        smtp_type = smtplib.SMTP_SSL if settings.SMTP_USE_SSL else smtplib.SMTP
        try:
            with smtp_type(self.host, settings.SMTP_PORT, timeout=30) as client:
                if settings.SMTP_USE_TLS and not settings.SMTP_USE_SSL:
                    client.starttls()
                if settings.SMTP_USERNAME:
                    client.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
                client.send_message(message)
        except Exception as exc:
            raise BusinessException(53002, f"邮件发送失败：{type(exc).__name__}") from exc
        return message.get("Message-ID")


def get_asr_provider() -> ASRProvider:
    if settings.MEETING_ASR_PROVIDER != "openai_compatible":
        raise BusinessException(52000, f"不支持的 ASR Provider：{settings.MEETING_ASR_PROVIDER}")
    return OpenAICompatibleASRProvider()


def get_email_provider() -> EmailProvider:
    if settings.MEETING_EMAIL_PROVIDER != "smtp":
        raise BusinessException(53000, f"不支持的邮件 Provider：{settings.MEETING_EMAIL_PROVIDER}")
    return SMTPEmailProvider()
