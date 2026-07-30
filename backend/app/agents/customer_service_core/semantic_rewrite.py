from __future__ import annotations

import re
from typing import Any

from backend.app.agents.customer_service_core.contextualizer import (
    explicit_order_ref,
    extract_budget_delta,
    extract_model,
    extract_order_fields,
    extract_price_max,
    extract_price_range,
    extract_product_attributes,
    extract_product_codes,
    extract_product_query_term,
    extract_target_references,
    requested_recommendation_count,
)
from backend.app.agents.customer_service_core.router import (
    deterministic_route,
    is_compare,
    is_handoff,
    is_logistics,
    is_product_search,
    is_recommend,
)
from backend.app.agents.customer_service_core.schemas import (
    ConversationDST,
    CustomerServiceIntent,
    CustomerServiceIntentClassification,
    OrderAction,
    ProductConstraintOperations,
    SlotOperation,
    SlotUpdate,
    domain_for_intent,
)
from backend.app.agents.customer_service_core.semantic_schemas import (
    AfterSalesSemanticPayload,
    FieldProposal,
    FieldSource,
    HandoffSemanticPayload,
    KnowledgeSemanticPayload,
    OrderSemanticPayload,
    ProductSemanticPayload,
    SemanticParseResult,
    TargetSemantics,
)
from backend.app.schemas.product import extract_product_category

_ORDINALS = {
    "first": 0,
    "top": 0,
    "former": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}


def parse(
    raw_query: str,
    dst: ConversationDST,
    messages: list[dict[str, Any]],
) -> SemanticParseResult:
    del dst, messages
    intent, _ = deterministic_route(raw_query)
    if intent == CustomerServiceIntent.OTHER:
        if is_compare(raw_query):
            intent = CustomerServiceIntent.PRODUCT_COMPARISON
        elif is_recommend(raw_query):
            intent = CustomerServiceIntent.PRODUCT_RECOMMENDATION
        elif is_product_search(raw_query):
            intent = CustomerServiceIntent.PRODUCT_SEARCH
    intent_proposal = (
        None
        if intent == CustomerServiceIntent.OTHER
        else FieldProposal(
            value=intent,
            source=(
                FieldSource.EXPLICIT
                if _has_explicit_intent_signal(raw_query, intent)
                else FieldSource.INFERRED
            ),
        )
    )
    domain_proposal = (
        None
        if intent_proposal is None
        else FieldProposal(
            value=domain_for_intent(intent),
            source=intent_proposal.source,
        )
    )
    target_semantics = _target_semantics(raw_query, intent)
    payload = _payload_for_query(raw_query, intent)
    gaps: list[str] = []
    if intent_proposal is None:
        gaps.append("intent_proposal")
    if domain_proposal is None:
        gaps.append("domain_proposal")
    if (
        intent
        in {
            CustomerServiceIntent.PRODUCT_REALTIME_FACT,
            CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
            CustomerServiceIntent.PRODUCT_COMPARISON,
        }
        and target_semantics is None
    ):
        gaps.append("target_semantics")
    if (
        isinstance(payload, ProductSemanticPayload)
        and intent
        in {
            CustomerServiceIntent.PRODUCT_REALTIME_FACT,
            CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        }
        and not payload.attributes
    ):
        gaps.append("payload_proposal")
    result = SemanticParseResult(
        intent_proposal=intent_proposal,
        domain_proposal=domain_proposal,
        target_semantics=target_semantics,
        payload_proposal=payload,
        gaps=gaps,
    )
    result.needs_llm = compute_needs_llm(result)
    return result


def from_classification(
    classification: CustomerServiceIntentClassification,
    raw_query: str,
) -> SemanticParseResult:
    intent = classification.intent
    return SemanticParseResult(
        intent_proposal=FieldProposal(value=intent, source=FieldSource.INFERRED),
        domain_proposal=FieldProposal(
            value=classification.domain or domain_for_intent(intent),
            source=FieldSource.INFERRED,
        ),
        target_semantics=_target_semantics_from_references(
            classification.target_references,
            raw_query,
            intent,
            source=FieldSource.INFERRED,
        ),
        payload_proposal=_payload_from_classification(classification, raw_query),
        gaps=[],
        conflicts=[],
        needs_llm=False,
    )


def merge(
    rule: SemanticParseResult,
    llm: SemanticParseResult,
) -> SemanticParseResult:
    result = rule.model_copy(deep=True)
    for field_name in ("intent_proposal", "domain_proposal"):
        rule_proposal = getattr(rule, field_name)
        llm_proposal = getattr(llm, field_name)
        if rule_proposal is None:
            setattr(result, field_name, llm_proposal)
        elif rule_proposal.source == FieldSource.EXPLICIT:
            continue
        elif llm_proposal is not None and rule_proposal.value != llm_proposal.value:
            if field_name not in result.conflicts:
                result.conflicts.append(field_name)
    result.target_semantics = _merge_target_semantics(
        rule.target_semantics,
        llm.target_semantics,
        conflicts=result.conflicts,
    )
    result.payload_proposal = _merge_payloads(
        rule.payload_proposal,
        llm.payload_proposal,
    )
    result.gaps = [gap for gap in result.gaps if not _gap_is_filled(result, gap)]
    result.needs_llm = False
    return result


def compute_needs_llm(value: SemanticParseResult) -> bool:
    if value.gaps or value.conflicts:
        return True
    return any(
        proposal is not None and proposal.source == FieldSource.INFERRED
        for proposal in (value.intent_proposal, value.domain_proposal)
    )


def _target_semantics(
    raw_query: str,
    intent: CustomerServiceIntent,
) -> TargetSemantics | None:
    return _target_semantics_from_references(
        extract_target_references(raw_query),
        raw_query,
        intent,
        source=FieldSource.EXPLICIT,
    )


def _target_semantics_from_references(
    references: list[str],
    raw_query: str,
    intent: CustomerServiceIntent,
    *,
    source: FieldSource,
) -> TargetSemantics | None:
    ordinals = [_ORDINALS[item] for item in references if item in _ORDINALS]
    if "bottom" in references or "latter" in references:
        ordinals.append(-1)
    chinese_ordinal = re.search(r"第([一二三四五六七八九十\\d]+)(?:个|款|笔)", raw_query)
    if chinese_ordinal is not None:
        number = _chinese_number(chinese_ordinal.group(1))
        if number is not None:
            ordinals.append(number - 1)
    ordinals = list(dict.fromkeys(ordinals))
    ordinal = ordinals[0] if ordinals else None
    order_ref = explicit_order_ref(raw_query)
    product_codes = extract_product_codes(raw_query)
    explicit_ref = (
        order_ref
        if intent
        in {
            CustomerServiceIntent.ORDER_QUERY,
            CustomerServiceIntent.LOGISTICS_QUERY,
            CustomerServiceIntent.AFTER_SALES,
            CustomerServiceIntent.HUMAN_HANDOFF,
        }
        else product_codes[0]
        if product_codes
        else None
    )
    semantic_reference_ids = [
        item
        for item in references
        if item not in _ORDINALS and item not in {"bottom", "latter", "it", "this", "that"}
    ]
    explicit_refs = list(
        dict.fromkeys(
            [value for value in [explicit_ref, *semantic_reference_ids] if value is not None]
        )
    )
    if explicit_ref is None and explicit_refs:
        explicit_ref = explicit_refs[0]
    model = extract_model(raw_query)
    category = extract_product_category(raw_query)
    reference_text = next(iter(references), None)
    if all(item is None for item in (ordinal, explicit_ref, model, category, reference_text)):
        return None
    return TargetSemantics(
        ordinal=ordinal,
        ordinals=ordinals,
        explicit_ref=explicit_ref,
        explicit_refs=explicit_refs,
        explicit_ref_source=(source if explicit_refs else None),
        explicit_name=model,
        category_hint=category,
        reference_text=reference_text,
    )


def _payload_for_query(
    raw_query: str,
    intent: CustomerServiceIntent,
):
    if intent in {
        CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        CustomerServiceIntent.PRODUCT_SEARCH,
        CustomerServiceIntent.PRODUCT_REALTIME_FACT,
        CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        CustomerServiceIntent.PRODUCT_COMPARISON,
    }:
        return ProductSemanticPayload(
            constraint_ops=_rule_constraint_operations(raw_query),
            attributes=extract_product_attributes(raw_query),
            recommendation_count=requested_recommendation_count(raw_query),
        )
    if intent in {
        CustomerServiceIntent.ORDER_QUERY,
        CustomerServiceIntent.LOGISTICS_QUERY,
    }:
        fields = extract_order_fields(raw_query) or {}
        return OrderSemanticPayload(
            action=_order_action(raw_query, intent),
            explicit_order_ref=explicit_order_ref(raw_query),
            explicit_phone_last4=fields.get("customer_phone_last4"),
        )
    if intent == CustomerServiceIntent.AFTER_SALES:
        return AfterSalesSemanticPayload(
            issue_type=_after_sales_issue_type(raw_query),
            order_ref=explicit_order_ref(raw_query),
        )
    if intent == CustomerServiceIntent.HUMAN_HANDOFF:
        return HandoffSemanticPayload(
            order_ref=explicit_order_ref(raw_query),
            reason="customer_request" if is_handoff(raw_query) else None,
        )
    if intent == CustomerServiceIntent.POLICY_QUESTION:
        return KnowledgeSemanticPayload(question=raw_query)
    return None


def _payload_from_classification(
    classification: CustomerServiceIntentClassification,
    raw_query: str,
):
    intent = classification.intent
    if intent in {
        CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        CustomerServiceIntent.PRODUCT_SEARCH,
        CustomerServiceIntent.PRODUCT_REALTIME_FACT,
        CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
        CustomerServiceIntent.PRODUCT_COMPARISON,
    }:
        return ProductSemanticPayload(
            constraint_ops=classification.constraint_operations,
            attributes=classification.attributes,
            recommendation_count=classification.recommendation_count,
        )
    if intent in {
        CustomerServiceIntent.ORDER_QUERY,
        CustomerServiceIntent.LOGISTICS_QUERY,
    }:
        try:
            action = OrderAction(classification.action) if classification.action else None
        except ValueError:
            action = None
        return OrderSemanticPayload(
            action=action,
            explicit_order_ref=explicit_order_ref(raw_query),
        )
    return _payload_for_query(raw_query, intent)


def _rule_constraint_operations(raw_query: str) -> ProductConstraintOperations:
    values: dict[str, SlotUpdate] = {}
    category = extract_product_category(raw_query)
    if category is not None:
        values["category"] = SlotUpdate(op=SlotOperation.SET, value=category)
    model = extract_model(raw_query)
    if model is not None:
        values["model"] = SlotUpdate(op=SlotOperation.SET, value=model)
    price_max = extract_price_max(raw_query)
    if price_max is not None:
        values["price_max"] = SlotUpdate(op=SlotOperation.SET, value=price_max)
    price_range = extract_price_range(raw_query)
    if price_range is not None:
        values["price_min"] = SlotUpdate(op=SlotOperation.SET, value=price_range[0])
        values["price_max"] = SlotUpdate(op=SlotOperation.SET, value=price_range[1])
    keyword = extract_product_query_term(raw_query)
    if (
        category is None
        and model is None
        and keyword is not None
        and _is_meaningful_keyword(keyword)
    ):
        values["keyword"] = SlotUpdate(op=SlotOperation.SET, value=keyword)
    if extract_budget_delta(raw_query) is not None:
        # Delta application needs trusted prior DST and remains in the validation layer.
        pass
    return ProductConstraintOperations.model_validate(values)


def _is_meaningful_keyword(value: str) -> bool:
    if any(character.isdigit() for character in value):
        return False
    return not any(
        token in value for token in ("容易", "适合", "预算", "区间", "个人用", "两个人用")
    )


def _order_action(
    raw_query: str,
    intent: CustomerServiceIntent,
) -> OrderAction:
    if "订单" in raw_query and any(word in raw_query for word in ("几个", "多少个", "多少笔")):
        return OrderAction.COUNT
    if "订单" in raw_query and is_compare(raw_query):
        return OrderAction.COMPARE
    if "订单" in raw_query and any(
        word in raw_query for word in ("我的订单", "订单列表", "所有订单", "全部订单")
    ):
        return OrderAction.LIST
    if intent == CustomerServiceIntent.LOGISTICS_QUERY or is_logistics(raw_query):
        return OrderAction.LOGISTICS
    return OrderAction.DETAIL


def _after_sales_issue_type(raw_query: str) -> str | None:
    for value in ("退货", "换货", "维修"):
        if value in raw_query:
            return value
    return None


def _chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return digits.get(value)


def _merge_target_semantics(
    rule: TargetSemantics | None,
    llm: TargetSemantics | None,
    *,
    conflicts: list[str],
) -> TargetSemantics | None:
    if rule is None:
        return llm
    if llm is None:
        return rule
    merged = rule.model_copy(deep=True)
    for field_name in TargetSemantics.model_fields:
        current = getattr(rule, field_name)
        proposed = getattr(llm, field_name)
        if current in (None, [], ""):
            setattr(merged, field_name, proposed)
        elif field_name == "reference_text":
            continue
        elif proposed is not None and current != proposed:
            marker = f"target_semantics.{field_name}"
            if marker not in conflicts:
                conflicts.append(marker)
    return merged


def _merge_payloads(rule, llm):
    if rule is None:
        return llm
    if llm is None or type(rule) is not type(llm):
        return rule
    merged = rule.model_copy(deep=True)
    for field_name in type(rule).model_fields:
        current = getattr(rule, field_name)
        proposed = getattr(llm, field_name)
        if isinstance(rule, ProductSemanticPayload) and field_name == "constraint_ops":
            merged.constraint_ops = _merge_constraint_operations(current, proposed)
            continue
        if (
            isinstance(rule, OrderSemanticPayload)
            and field_name == "action"
            and proposed is not None
        ):
            merged.action = proposed
            continue
        if current in (None, [], "") and proposed not in (None, [], ""):
            setattr(merged, field_name, proposed)
    return merged


def _merge_constraint_operations(
    rule: ProductConstraintOperations,
    llm: ProductConstraintOperations,
) -> ProductConstraintOperations:
    values: dict[str, SlotUpdate] = {}
    for field_name in ProductConstraintOperations.model_fields:
        rule_update = getattr(rule, field_name)
        llm_update = getattr(llm, field_name)
        values[field_name] = rule_update if rule_update.op != SlotOperation.KEEP else llm_update
    return ProductConstraintOperations.model_validate(values)


def _has_explicit_intent_signal(
    raw_query: str,
    intent: CustomerServiceIntent,
) -> bool:
    if intent == CustomerServiceIntent.GREETING:
        return raw_query.strip() in {"你好", "您好", "hello", "hi"}
    if intent in {
        CustomerServiceIntent.PRODUCT_RECOMMENDATION,
        CustomerServiceIntent.PRODUCT_SEARCH,
        CustomerServiceIntent.PRODUCT_COMPARISON,
    }:
        return is_recommend(raw_query) or is_product_search(raw_query) or is_compare(raw_query)
    if intent in {
        CustomerServiceIntent.PRODUCT_REALTIME_FACT,
        CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
    }:
        return bool(extract_product_attributes(raw_query))
    if intent == CustomerServiceIntent.ORDER_QUERY:
        return "订单" in raw_query
    if intent == CustomerServiceIntent.LOGISTICS_QUERY:
        return is_logistics(raw_query)
    if intent == CustomerServiceIntent.AFTER_SALES:
        return any(value in raw_query for value in ("退货", "换货", "维修", "售后"))
    if intent == CustomerServiceIntent.POLICY_QUESTION:
        return True
    return False


def _gap_is_filled(value: SemanticParseResult, gap: str) -> bool:
    if gap == "payload_proposal":
        return value.payload_proposal is not None
    return getattr(value, gap, None) is not None
