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
    business_tool_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    knowledge_data = knowledge if isinstance(knowledge, dict) else {}
    sources = _list_value(knowledge_data.get("sources"))
    citations = _list_value(knowledge_data.get("citations"))
    business_evidence = (
        business_tool_evidence if isinstance(business_tool_evidence, list) else []
    )
    source_count = len(sources)
    citation_count = len(citations)
    business_evidence_count = len(business_evidence)
    retrieval_evidence_count = max(source_count, citation_count)
    evidence_count = retrieval_evidence_count + business_evidence_count
    retrieval_used = bool(knowledge_data)
    if retrieval_evidence_count > 0:
        evidence_type = "rag"
        evidence_sources = ["knowledge_search"]
    elif business_evidence_count > 0:
        evidence_type = "business_tool"
        evidence_sources = sorted(
            {
                str(item.get("tool_name"))
                for item in business_evidence
                if isinstance(item, dict) and item.get("tool_name")
            }
        )
    else:
        evidence_type = "none"
        evidence_sources = []
    return {
        "retrieval_used": retrieval_used,
        "knowledge_base_id": knowledge_base_id,
        "evidence_count": evidence_count,
        "evidence_type": evidence_type,
        "evidence_sources": evidence_sources,
        "business_evidence_count": business_evidence_count,
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


def collect_customer_service_business_evidence(tool_results: Any) -> list[dict[str, Any]]:
    trusted_tools = {
        "search_products",
        "recommend_products",
        "compare_products",
        "query_order",
        "query_logistics",
        "create_after_sales_ticket",
        "create_human_handoff",
    }
    results = tool_results if isinstance(tool_results, list) else []
    evidence: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict) or item.get("success") is not True:
            continue
        tool_name = item.get("tool_name")
        if tool_name not in trusted_tools:
            continue
        if not _has_meaningful_business_result(item.get("result")):
            continue
        evidence.append({"tool_name": tool_name})
    return evidence


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


def _has_meaningful_business_result(value: Any) -> bool:
    if isinstance(value, dict):
        items = value.get("items")
        if isinstance(items, list):
            return bool(items)
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value not in {None, ""}
