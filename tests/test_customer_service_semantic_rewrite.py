from backend.app.agents.customer_service_core.resolver import resolve_target
from backend.app.agents.customer_service_core.schemas import (
    CandidateHistoryEntry,
    CandidateRef,
    ConversationDST,
    CustomerServiceDomain,
    CustomerServiceIntent,
    CustomerServiceIntentClassification,
    DomainState,
)
from backend.app.agents.customer_service_core.semantic_rewrite import (
    from_classification,
    merge,
    parse,
)
from backend.app.agents.customer_service_core.semantic_schemas import FieldSource


def test_rule_parse_extracts_semantics_without_resolving_target() -> None:
    result = parse(
        "第一款鼠标支持蓝牙吗",
        ConversationDST(),
        [],
    )

    assert result.intent == CustomerServiceIntent.PRODUCT_DOCUMENT_FACT
    assert result.intent_proposal is not None
    assert result.intent_proposal.source == FieldSource.EXPLICIT
    assert result.target_semantics is not None
    assert result.target_semantics.ordinal == 0
    assert result.target_semantics.category_hint is not None
    assert result.needs_llm is False


def test_resolver_uses_matching_category_history_order() -> None:
    dst = ConversationDST(
        domains={
            CustomerServiceDomain.PRODUCT: DomainState(
                candidates=[
                    CandidateRef(ref="KEYBOARD-1", display_name="键盘", position=1),
                ],
                candidate_history=[
                    CandidateHistoryEntry(
                        ref="MOUSE-OLD",
                        display_name="旧鼠标",
                        category="鼠标和指针设备",
                        batch_id="batch-1",
                        position=1,
                    ),
                    CandidateHistoryEntry(
                        ref="MOUSE-NEW",
                        display_name="新鼠标",
                        category="鼠标和指针设备",
                        batch_id="batch-2",
                        position=1,
                    ),
                    CandidateHistoryEntry(
                        ref="KEYBOARD-1",
                        display_name="键盘",
                        category="键盘",
                        batch_id="batch-3",
                        position=1,
                    ),
                ],
            )
        }
    )
    semantics = parse("第一款鼠标支持蓝牙吗", dst, []).target_semantics

    resolution = resolve_target(
        intent=CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        semantics=semantics,
        dst=dst,
    )

    assert resolution.resolved_ids == ["MOUSE-OLD"]


def test_targetless_recommendation_does_not_require_clarification() -> None:
    resolution = resolve_target(
        intent=CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        semantics=None,
        dst=ConversationDST(),
    )

    assert resolution.clarification_required is False
    assert resolution.resolved_ids == []


def test_order_list_does_not_require_selected_order() -> None:
    resolution = resolve_target(
        intent=CustomerServiceIntent.ORDER_QUERY,
        semantics=None,
        dst=ConversationDST(),
    )

    assert resolution.clarification_required is False


def test_llm_conflict_is_retained_for_clarification() -> None:
    rule = parse("客服电话多少", ConversationDST(), [])
    llm = from_classification(
        CustomerServiceIntentClassification(
            intent=CustomerServiceIntent.POLICY_QUESTION,
            confidence=0.9,
        ),
        "客服电话多少",
    )

    result = merge(rule, llm)

    assert "intent_proposal" in result.conflicts


def test_llm_can_fill_missing_target_without_creating_false_conflict() -> None:
    rule = parse("这款鼠标支持蓝牙吗", ConversationDST(), [])
    llm = from_classification(
        CustomerServiceIntentClassification(
            intent=CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
            confidence=0.9,
            target_references=["second"],
            attributes=["bluetooth"],
        ),
        "这款鼠标支持蓝牙吗",
    )

    result = merge(rule, llm)

    assert result.target_semantics is not None
    assert result.target_semantics.ordinal == 1
    assert result.conflicts == []
