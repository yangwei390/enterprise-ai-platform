from backend.app.agents.evidence import (
    build_evidence_metadata,
    collect_customer_service_business_evidence,
)


def test_successful_customer_service_tool_is_business_evidence() -> None:
    business_evidence = collect_customer_service_business_evidence(
        [
            {
                "tool_name": "recommend_products",
                "success": True,
                "result": {"items": [{"product_code": "3"}]},
            }
        ]
    )

    metadata = build_evidence_metadata(
        knowledge=None,
        knowledge_base_id=None,
        retrieval_required=False,
        business_tool_evidence=business_evidence,
    )

    assert metadata["evidence_type"] == "business_tool"
    assert metadata["evidence_sources"] == ["recommend_products"]
    assert metadata["business_evidence_count"] == 1
    assert metadata["grounded_answer"] is True
    assert metadata["no_evidence"] is False


def test_empty_customer_service_tool_result_is_not_evidence() -> None:
    business_evidence = collect_customer_service_business_evidence(
        [
            {
                "tool_name": "recommend_products",
                "success": True,
                "result": {"items": []},
            }
        ]
    )

    metadata = build_evidence_metadata(
        knowledge=None,
        knowledge_base_id=None,
        retrieval_required=False,
        business_tool_evidence=business_evidence,
    )

    assert metadata["evidence_type"] == "none"
    assert metadata["evidence_count"] == 0
    assert metadata["grounded_answer"] is False


def test_customer_service_observation_raw_result_is_business_evidence() -> None:
    business_evidence = collect_customer_service_business_evidence(
        [
            {
                "tool_name": "recommend_products",
                "success": True,
                "raw_result": {"items": [{"product": {"product_code": "3"}}]},
            }
        ]
    )

    assert business_evidence == [{"tool_name": "recommend_products"}]
