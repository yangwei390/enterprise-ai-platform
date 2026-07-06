# Coding Rules

## Quality Gates

Python changes must pass:

- `pyright`
- `ruff check .`
- `python3.12 -m compileall backend/app`

When the task names specific files, also run the requested `py_compile` command.

## Scope Control

- Confirm the task scope before changing code.
- Do not modify unrelated files.
- Do not add hidden features or opportunistic refactors.
- Do not break existing architecture to solve a small local issue.
- Prefer narrow, traceable changes.

## Architecture Rules

- New modules should keep a single responsibility.
- API layer must not bypass Service / Repository / ORM boundaries.
- Business layers must not directly depend on vendor SDKs.
- Vendor SDK calls belong in Client implementations.
- Provider coordinates model semantics.
- Factory selects implementations.

## API Rules

- API responses must use unified `ApiResponse`.
- Existing API paths must not change unless explicitly requested.
- Existing response compatibility must be preserved when possible.

## Configuration Rules

- App configuration goes through `settings`.
- LLM configuration goes through `LLMConfig`.
- Do not hardcode API keys, passwords, tokens, hosts, or credentials.
- Sensitive information belongs only in `.env`.
- `.env.example` is a template only.
- `.env.example` must not contain real secrets.
- If `.env` already exists, never overwrite it with `cp .env.example .env`.

## Validation Discipline

- Run validation after edits.
- Do not bypass errors by commenting code or suppressing checks without cause.
- Fix root causes.
- If a validation failure is unrelated to the task, report it clearly.
