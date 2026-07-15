from typing import Any

NO_EVIDENCE_ANSWER = "无法基于当前知识库证据回答该问题。"


def requires_evidence(retrieval_policy: dict | None) -> bool:
    if not isinstance(retrieval_policy, dict):
        return False
    return bool(
        retrieval_policy.get("required")
        or retrieval_policy.get("require_evidence")
    )


def build_evidence_metadata(
    *,
    knowledge: dict | None,
    knowledge_base_id: int | None,
    retrieval_required: bool,
) -> dict[str, Any]:
    knowledge_data = knowledge if isinstance(knowledge, dict) else {}
    sources = _list_value(knowledge_data.get("sources"))
    citations = _list_value(knowledge_data.get("citations"))
    source_count = len(sources)
    citation_count = len(citations)
    evidence_count = max(source_count, citation_count)
    retrieval_used = bool(knowledge_data)
    return {
        "retrieval_used": retrieval_used,
        "knowledge_base_id": knowledge_base_id,
        "evidence_count": evidence_count,
        "source_count": source_count,
        "citation_count": citation_count,
        "selected_document_ids": selected_document_ids(sources=sources, citations=citations),
        "no_evidence": retrieval_required and evidence_count == 0,
        "grounded_answer": evidence_count > 0,
        "errors": [],
    }


def has_evidence(knowledge: dict | None) -> bool:
    metadata = build_evidence_metadata(
        knowledge=knowledge,
        knowledge_base_id=None,
        retrieval_required=False,
    )
    return bool(metadata["evidence_count"] > 0)


def selected_document_ids(*, sources: list, citations: list) -> list[int]:
    return sorted(
        {
            document_id
            for item in [*sources, *citations]
            if isinstance(item, dict)
            for document_id in [_optional_int(item.get("document_id"))]
            if document_id is not None
        }
    )


def _list_value(value: Any) -> list:
    return value if isinstance(value, list) else []


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
