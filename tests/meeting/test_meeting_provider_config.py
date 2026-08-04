import pytest
from backend.app.config.settings import settings
from backend.app.exceptions import BusinessException
from backend.app.meetings.providers import OpenAICompatibleASRProvider, SMTPEmailProvider


def test_asr_missing_credentials_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "MEETING_ASR_API_KEY", None)
    with pytest.raises(BusinessException, match="MEETING_ASR_API_KEY"):
        OpenAICompatibleASRProvider()


def test_email_missing_credentials_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "SMTP_HOST", None)
    monkeypatch.setattr(settings, "SMTP_FROM_ADDRESS", None)
    with pytest.raises(BusinessException, match="SMTP_HOST"):
        SMTPEmailProvider()
