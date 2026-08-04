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
