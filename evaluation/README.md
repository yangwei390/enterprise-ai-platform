# Enterprise RAG Evaluation

Lightweight evaluation for development-time regression checks.

## Scope

Evaluation is independent from the Retriever Pipeline and does not participate in:

- `/chat`
- `/debug/rag-trace`
- `/debug/retriever-compare`

It reuses the existing `ChatService` and evaluates the returned answer, sources, and metadata.

## Dataset

Questions live in:

```text
evaluation/datasets/questions.yaml
```

Supported fields:

- `id`
- `question`
- `expected_documents`
- `expected_chunks`
- `expected_keywords`
- `knowledge_base_id`
- `top_k`
- `score_threshold`

## Run

```bash
python -m evaluation.run
```

The command writes:

```text
evaluation/report.json
```
