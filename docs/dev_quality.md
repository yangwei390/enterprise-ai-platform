# Development Quality Gates

This project uses Python static checks and pre-commit hooks to catch basic errors before code reaches the server.

## Install

```bash
pip install -r requirements.txt
pre-commit install
```

## Manual Checks

```bash
ruff check backend
black --check backend
pyright backend
python -m compileall backend/app
```

## Auto Fix

```bash
ruff check backend --fix
black backend
```

## Commit

```bash
git add .
git commit -m "xxx"
```

## Tools

- `ruff`: fast linting for import errors, unused variables, style problems, and common bug patterns.
- `black`: deterministic Python formatting.
- `pyright`: static type checking.
- `pytest`: test discovery and execution.
- `pre-commit`: runs quality gates before Git commits.
