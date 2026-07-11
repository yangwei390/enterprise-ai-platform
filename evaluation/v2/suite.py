import ast
import json
from pathlib import Path
from typing import Any

from evaluation.v2.errors import EvaluationSuiteError
from evaluation.v2.schemas import EvaluationCase, EvaluationSuite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"


def load_suite(path: str | Path) -> EvaluationSuite:
    suite_path = _safe_eval_path(path)
    raw = _load_mapping(suite_path)
    return EvaluationSuite.model_validate(raw)


def load_v1_questions_as_suite(path: Path) -> EvaluationSuite:
    from evaluation.run import load_questions

    questions = load_questions(path)
    cases = [
        EvaluationCase(
            id=question.id,
            name=question.question,
            target="rag",
            query=question.question,
            input={
                "knowledge_base_id": question.knowledge_base_id,
                "top_k": question.top_k,
                "score_threshold": question.score_threshold,
                "mode": "rag-answer",
            },
            expected={
                "documents": question.expected_documents,
                "chunk_indexes": question.expected_chunks,
                "keywords": question.expected_keywords,
            },
            metrics=[
                "retriever_hit",
                "chunk_recall",
                "keyword_coverage",
                "latency_ms",
            ],
            thresholds={"keyword_coverage": {"min": 0.5}, "retriever_hit": True},
        )
        for question in questions
    ]
    return EvaluationSuite(
        id="evaluation_v1_compat",
        name="Evaluation V1 Compatibility",
        version="1.0",
        cases=cases,
        default_metrics=[],
        default_thresholds={},
    )


def list_available_suites() -> list[dict[str, Any]]:
    fixture_dir = EVALUATION_ROOT / "v2" / "fixtures" / "suites"
    suites = []
    for path in sorted([*fixture_dir.glob("*.yaml"), *fixture_dir.glob("*.json")]):
        try:
            suite = load_suite(path)
            suites.append(
                {
                    "id": suite.id,
                    "name": suite.name,
                    "version": suite.version,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "case_count": len(suite.cases),
                }
            )
        except Exception:
            continue
    return suites


def _safe_eval_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if EVALUATION_ROOT.resolve() not in [resolved, *resolved.parents]:
        raise EvaluationSuiteError("suite path must be under evaluation/")
    if not resolved.exists():
        raise EvaluationSuiteError(f"suite file not found: {resolved}")
    return resolved


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise EvaluationSuiteError("suite root must be object")
        return data
    except ModuleNotFoundError:
        return _load_limited_yaml(text)


def _load_limited_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_by_indent: dict[int, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            item_text = line[2:]
            if not isinstance(parent, list):
                key = last_key_by_indent.get(indent - 2)
                container = stack[-1][1]
                if isinstance(container, dict) and key:
                    container[key] = []
                    parent = container[key]
                    stack.append((indent - 2, parent))
            if ":" in item_text:
                key, value = _split(item_text)
                item: dict[str, Any] = {key: _parse_scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_text))
            continue
        key, value = _split(line)
        if value == "":
            container: dict[str, Any] | list[Any]
            container = [] if key in {"cases", "metrics", "tags", "requires"} else {}
            parent[key] = container
            last_key_by_indent[indent] = key
            stack.append((indent, container))
        else:
            parent[key] = _parse_scalar(value)
            last_key_by_indent[indent] = key
    return root


def _split(text: str) -> tuple[str, str]:
    key, _, value = text.partition(":")
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() == "null":
        return None
    if value.startswith("[") or value.startswith("{"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")
