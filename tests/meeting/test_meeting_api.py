from backend.app.api.meeting import get_service, router
from backend.app.exceptions import register_exception_handlers
from backend.app.meetings.repository import MeetingRepository
from backend.app.meetings.service import MeetingService
from backend.app.models import Base
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool


def test_create_list_and_owner_isolation_api():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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
    service = MeetingService(MeetingRepository(db))
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    created = client.post(
        "/meetings",
        headers={"X-Owner-Id": "owner-a"},
        json={"title": "API sync", "participants": ["Alice"]},
    ).json()
    assert created["code"] == 0
    meeting_id = created["data"]["id"]
    listed = client.get("/meetings", headers={"X-Owner-Id": "owner-a"}).json()
    assert listed["data"]["total"] == 1
    forbidden = client.get(f"/meetings/{meeting_id}", headers={"X-Owner-Id": "owner-b"}).json()
    assert forbidden["code"] == 40420
    missing_header = client.get("/meetings")
    assert missing_header.status_code == 422
    db.close()
