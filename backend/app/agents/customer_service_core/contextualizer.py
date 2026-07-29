from __future__ import annotations

import re
from collections.abc import Mapping

from backend.app.agents.customer_service_core.router import classify_user_decision
from backend.app.agents.customer_service_core.schemas import (
    AfterSalesPayload,
    ContextualizedRequest,
    CustomerServiceIntent,
    CustomerServiceIntentMode,
    CustomerServiceRecognitionSource,
    CustomerServiceSource,
    GeneralPayload,
    HandoffPayload,
    KnowledgePayload,
    OrderPayload,
    ProductPayload,
    ProductRequestConstraints,
    TargetResolutionSource,
    UserDecision,
    action_for_intent,
    domain_for_intent,
)


def extract_order_fields(query: str) -> dict[str, str] | None:
    order_match = re.search(r"\b(\d{10,20})\b", query)
    last4_match = re.search(r"(?:后四位|尾号|手机号后四位)\D*(\d{4})", query)
    if order_match is None or last4_match is None:
        return None
    return {
        "order_no": order_match.group(1),
        "customer_phone_last4": last4_match.group(1),
    }


def explicit_order_ref(query: str) -> str | None:
    explicit = re.search(r"\b(\d{10,20})\b", query)
    if explicit is not None:
        return explicit.group(1)
    masked = re.search(r"\b(\d{4}\*{4}\d{4})\b", query)
    return masked.group(1) if masked is not None else None


def extract_product_codes(query: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for code in re.findall(r"\b[A-Z]{1,6}\d{2,12}\b", query):
        if code not in seen:
            result.append(code)
            seen.add(code)
    return result


def extract_model(query: str) -> str | None:
    explicit = re.search(r"型号[:：\s]*([A-Za-z0-9_-]{2,64})", query)
    if explicit:
        return explicit.group(1)
    model = re.search(r"\b([A-Z]{1,6}\d{2,12})\b", query)
    return model.group(1) if model else None


def extract_price_max(query: str) -> int | None:
    match = re.search(r"(\d{2,6})\s*(?:以内|以下|内)", query)
    if match is None:
        match = re.search(r"预算(?:是|为|到|提高到|调整到)?\s*(\d{2,6})", query)
    return int(match.group(1)) if match else None


def extract_budget_delta(query: str) -> int | None:
    common_typo = re.search(r"再\s*长\s*(\d{1,6})\s*预算", query)
    if common_typo is not None:
        return int(common_typo.group(1))
    increase = re.search(
        r"(?:预算\s*)?(?:再\s*)?(?:加|增加|提高|上调|涨)\s*(\d{1,6})",
        query,
    )
    if increase is not None:
        return int(increase.group(1))
    decrease = re.search(
        r"(?:预算\s*)?(?:再\s*)?(?:减|减少|降低|下调|降)\s*(\d{1,6})",
        query,
    )
    return -int(decrease.group(1)) if decrease is not None else None


def extract_price_range(query: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{1,6})\s*(?:到|至|[-~～])\s*(\d{1,6})", query)
    if match is None:
        return None
    lower, upper = int(match.group(1)), int(match.group(2))
    return (lower, upper) if lower <= upper else (upper, lower)


def extract_target_references(query: str) -> list[str]:
    result: list[str] = []
    labels = ("first", "second", "third", "fourth", "fifth")
    for index, pattern in enumerate(
        (
            re.compile(r"第\s*一(?:个|款|件|只)?"),
            re.compile(r"第\s*二(?:个|款|件|只)?"),
            re.compile(r"第\s*三(?:个|款|件|只)?"),
            re.compile(r"第\s*四(?:个|款|件|只)?"),
            re.compile(r"第\s*五(?:个|款|件|只)?"),
        )
    ):
        if pattern.search(query):
            result.append(labels[index])
    for reference, phrases in {
        "top": ("上面那款", "上面那个", "上面的", "前者"),
        "bottom": ("下面那款", "下面那个", "下面的", "后者"),
    }.items():
        if any(phrase in query for phrase in phrases):
            result.append(reference)
    for product_code in extract_product_codes(query):
        if product_code not in result:
            result.append(product_code)
    return result[:5]


def extract_product_attributes(query: str) -> list[str]:
    attribute_words = {
        "price": ("价格", "多少钱"),
        "inventory": ("库存", "有货"),
        "brand": ("品牌",),
        "sale_status": ("在售", "销售状态"),
        "dimensions": ("尺寸", "大小", "长宽高"),
        "weight": ("重量", "多重"),
        "button_count": ("按键", "几个键"),
        "package_contents": ("包装", "盒内", "配件"),
        "connection": ("连接", "配对"),
        "bluetooth": ("蓝牙",),
        "compatibility": ("兼容", "系统支持"),
        "charging": ("充电", "电池"),
        "operation": ("怎么用", "使用", "操作"),
        "troubleshooting": ("故障", "失灵", "没反应"),
    }
    return [
        attribute
        for attribute, words in attribute_words.items()
        if any(word in query for word in words)
    ][:5]


def requested_recommendation_count(query: str) -> int | None:
    match = re.search(
        r"(?:推荐|介绍|选|找)[^，。！？]{0,8}?"
        r"(?P<count>\d+|[一二两三四五六七八九十]+)\s*(?:个|款|件|只)",
        query,
    )
    if match is None:
        return None
    value = match.group("count")
    if value.isdigit():
        return int(value)
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" in value:
        tens, ones = value.split("十", 1)
        return digits.get(tens, 1) * 10 + digits.get(ones, 0)
    return digits.get(value)


def extract_product_query_term(query: str) -> str | None:
    value = query.strip()
    if not value or len(value) > 256:
        return None
    value = re.sub(
        r"^(?:请|麻烦)?\s*(?:给我|帮我|替我|为我)?\s*"
        r"(?:推荐|介绍|找|查找|搜索|查询|查(?:一下)?|看看|选|挑|"
        r"我要|我想要|我需要|需要)\s*",
        "",
        value,
    )
    value = re.sub(
        r"^(?:一|二|两|三|四|五|几|\d+)\s*(?:个|款|件|只|台|套|把|副)?\s*",
        "",
        value,
    )
    value = re.sub(r"^(?:在售|有货)\s*的?\s*", "", value)
    value = re.sub(
        r"(?:给我|推荐)?\s*[吧吗呢么]?[？?！!。]*$",
        "",
        value,
    ).strip()
    value = re.sub(r"^(?:那么|那就|那(?:个|款|件|种)?)\s*", "", value).strip()
    if not value or value == query.strip():
        return None
    if value in {"一个", "一款", "商品", "产品", "其他", "别的"}:
        return None
    return value[:128]


def rewrite_contextual_query(
    raw_query: str,
    *,
    intent: CustomerServiceIntent,
    target_product_codes: list[str],
    attributes: list[str],
    constraints: ProductRequestConstraints,
) -> str:
    parts = [f"意图={intent.value}"]
    if target_product_codes:
        parts.append(f"商品={','.join(target_product_codes)}")
    if attributes:
        parts.append(f"属性={','.join(attributes)}")
    values = {
        key: value
        for key, value in constraints.model_dump(exclude_none=True).items()
        if value not in ([], "")
    }
    if values:
        parts.append(f"约束={values!r}")
    parts.append(f"用户请求={raw_query}")
    return "；".join(parts)


def build_contextualized_request(
    *,
    raw_query: str,
    intent: CustomerServiceIntent,
    source: CustomerServiceSource,
    intent_mode: CustomerServiceIntentMode,
    classifier: str,
    confidence: float | None,
    target_references: list[str],
    target_product_codes: list[str],
    attributes: list[str],
    recommendation_count: int | None,
    product_resolution_source: TargetResolutionSource,
    constraints: ProductRequestConstraints,
    order_payload: OrderPayload | None,
    identity_fields: Mapping[str, object],
    pending_after_sales: Mapping[str, object],
    rewritten_query: str | None,
    clarification_question: str | None,
) -> ContextualizedRequest:
    product_payload = (
        ProductPayload(
            target_product_codes=target_product_codes,
            attributes=attributes,
            recommendation_count=recommendation_count,
            resolution_source=product_resolution_source,
            constraints=constraints,
        )
        if intent
        in {
            CustomerServiceIntent.PRODUCT_RECOMMENDATION,
            CustomerServiceIntent.PRODUCT_SEARCH,
            CustomerServiceIntent.PRODUCT_REALTIME_FACT,
            CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
            CustomerServiceIntent.PRODUCT_COMPARISON,
        }
        else None
    )
    payload = (
        product_payload
        or order_payload
        or _non_transactional_payload(
            raw_query=raw_query,
            intent=intent,
            identity_fields=identity_fields,
            pending_after_sales=pending_after_sales,
        )
    )
    recognition_source = (
        CustomerServiceRecognitionSource.LLM
        if classifier == "llm"
        else CustomerServiceRecognitionSource.FALLBACK
        if classifier in {"rules_fallback", "llm_failed"}
        else CustomerServiceRecognitionSource.RULES
    )
    return ContextualizedRequest(
        raw_query=raw_query,
        rewritten_query=rewritten_query
        or rewrite_contextual_query(
            raw_query,
            intent=intent,
            target_product_codes=target_product_codes,
            attributes=attributes,
            constraints=constraints,
        ),
        domain=domain_for_intent(intent),
        intent=intent,
        action=(
            order_payload.action.value if order_payload is not None else action_for_intent(intent)
        ),
        source=source,
        intent_mode=intent_mode,
        recognition_source=recognition_source,
        target_references=target_references,
        payload=payload,
        confidence=confidence if confidence is not None else 1,
        clarification_required=clarification_question is not None,
        clarification_question=clarification_question,
    )


def _non_transactional_payload(
    *,
    raw_query: str,
    intent: CustomerServiceIntent,
    identity_fields: Mapping[str, object],
    pending_after_sales: Mapping[str, object],
) -> KnowledgePayload | AfterSalesPayload | HandoffPayload | GeneralPayload | None:
    if intent == CustomerServiceIntent.POLICY_QUESTION:
        return KnowledgePayload(question=raw_query)
    if intent == CustomerServiceIntent.HUMAN_HANDOFF:
        return HandoffPayload(
            order_ref=_string_value(identity_fields.get("order_no")),
            customer_phone_last4=_string_value(identity_fields.get("customer_phone_last4")),
            reason="customer_request",
            message=raw_query[:500],
        )
    if intent == CustomerServiceIntent.AFTER_SALES:
        return AfterSalesPayload(
            order_ref=_string_value(identity_fields.get("order_no")),
            customer_phone_last4=_string_value(identity_fields.get("customer_phone_last4")),
            issue_type=(
                _string_value(pending_after_sales.get("issue_type"))
                or (
                    "refund" if "退" in raw_query else "exchange" if "换" in raw_query else "repair"
                )
            ),
            reason=(_string_value(pending_after_sales.get("summary")) or raw_query or None),
            confirmed=classify_user_decision(raw_query) == UserDecision.CONFIRM,
        )
    if intent in {
        CustomerServiceIntent.GREETING,
        CustomerServiceIntent.OUT_OF_SCOPE,
        CustomerServiceIntent.OTHER,
    }:
        return GeneralPayload(topic=raw_query or None)
    return None


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
