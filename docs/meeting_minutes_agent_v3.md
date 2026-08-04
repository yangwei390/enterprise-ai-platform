# Meeting Minutes Agent V3

## Incremental architecture rules

- `backend/app/meetings/` owns meeting repositories, services, providers and LangGraph orchestration.
- Vendor calls stay behind provider interfaces; API handlers never call vendors or ORM directly.
- `backend/app/models/meeting.py` contains only meeting-domain persistence models.
- `frontend-web/src/pages/MeetingsPage.tsx` and `frontend-web/src/api/meetings.ts` are the V3 UI boundary.
- `tests/meeting/` contains injected fakes; runtime factories never expose fake ASR or email providers.
- Every read is scoped by the caller owner identifier until V3 gains authentication middleware.

The workflow uses the installed LangGraph runtime primitives for transcript chunk summarization,
topic merging, decision/action extraction and final minutes. Review approval and email confirmation
remain domain state transitions because they must be transactionally bound to persisted versions.

Migration revision follows V3 head `6d4e9a2f1b8c`. Existing V3 agent, RAG, customer-service and
workflow APIs remain unchanged.

## Run and acceptance

1. Add real `LLM_*`, `MEETING_ASR_*` and `SMTP_*` values to the local `.env`; never commit it.
2. Run `alembic upgrade head` and start FastAPI normally.
3. Run `npm ci && npm run dev` in `frontend-web`, then open `/meetings`.
4. Create a meeting, upload MP3/WAV/M4A, process it, seek evidence segments, bind speaker names,
   edit and approve the minutes, generate/edit an email draft, then explicitly confirm sending.

All `/meetings` endpoints require `X-Owner-Id`. Resources include list/detail, audio upload/playback,
process/retry/SSE events, transcript/minutes, speaker mapping, version approval, email draft editing,
confirmed send and email retry. Runtime has no fake-provider selection: missing credentials return
actionable configuration errors.

## State and safety

States are `uploaded`, `transcribing`, `generating_minutes`, `draft_ready`, `reviewing`, `approved`,
`email_draft_ready`, `sending`, `sent`, and `failed`. Approval is bound to a minutes version and
content revision. Editing approved content creates a reviewing version. Sending revalidates current
approval, recipients, non-empty content, expected draft revision and explicit confirmation. Delivery
attempts persist provider, final content, timestamps and errors without storing SMTP credentials.

## Provider replacement

Implement `ASRProvider.transcribe` or `EmailProvider.send`, add settings and factory selection, and
keep vendor calls inside `providers.py`. ASR normalization must return stable Speaker aliases,
timestamps, text, confidence and sanitized audit metadata. Speaker aliases never assert identity.

## Known V3 limitations

- `X-Owner-Id` is an isolation seam, not authentication; integrate V3 auth before public exposure.
- Processing uses FastAPI in-process background tasks, not a durable distributed queue.
- Audio duration comes from ASR metadata or the final segment end time.
- SMTP acceptance does not guarantee inbox delivery.
- Real ASR and email calls require the single remaining external step: valid provider credentials.

## Ten files to read

1. `frontend-web/src/pages/MeetingsPage.tsx`
2. `frontend-web/src/api/meetings.ts`
3. `backend/app/api/meeting.py`
4. `backend/app/meetings/service.py`
5. `backend/app/meetings/workflow.py`
6. `backend/app/meetings/providers.py`
7. `backend/app/meetings/repository.py`
8. `backend/app/models/meeting.py`
9. `alembic/versions/e8a1b2c3d4f5_create_meeting_minutes_agent_tables.py`
10. `tests/meeting/test_meeting_main_flow.py`
