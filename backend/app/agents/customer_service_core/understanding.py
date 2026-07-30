from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    ReferenceExpression,
    SemanticFrame,
)
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.schemas.product import normalize_product_category
from pydantic import BaseModel, ConfigDict, Field


class UnderstandingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: SemanticFrame
    rule_frame: SemanticFrame
    llm_frame: SemanticFrame | None = None
    llm_used: bool = False
    merge: dict[str, Any] = Field(default_factory=dict)


_INJECTION_TERMS = (
    "忽略系统",
    "忽略之前",
    "绕过规则",
    "system prompt",
    "开发者指令",
)
_MANUAL_TERMS = (
    "说明书",
    "怎么用",
    "如何使用",
    "能充电",
    "充电吗",
    "蓝牙",
    "按键",
    "连接",
    "兼容",
)
_PRICE_TERMS = ("多少钱", "价格", "售价")
_CONTINUATION_TERMS = (
    "还有其他",
    "还有别的",
    "还有吗",
    "换一个",
    "换一款",
    "再来一个",
    "再推荐",
    "不要这个",
)
_ORDINAL_RE = re.compile(r"第\s*([一二三四五1-5])\s*(?:个|款|件|只|笔)?")
_ORDINALS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4}


async def understand(
    *,
    query: str,
    state: CustomerServiceState,
    messages: list[dict[str, Any]],
) -> UnderstandingResult:
    rule_frame = _rule_understand(query, state)
    if rule_frame.intent != "other":
        return UnderstandingResult(
            frame=rule_frame,
            rule_frame=rule_frame,
            merge={"source": "rules", "conflicts": []},
        )
    llm_frame = await _llm_understand(query, state, messages)
    merged = _merge(rule_frame, llm_frame)
    return UnderstandingResult(
        frame=merged,
        rule_frame=rule_frame,
        llm_frame=llm_frame,
        llm_used=True,
        merge={"source": "llm_fallback", "conflicts": []},
    )


def _rule_understand(query: str, state: CustomerServiceState) -> SemanticFrame:
    normalized = query.strip()
    lowered = normalized.casefold()
    if any(term in lowered for term in _INJECTION_TERMS):
        return SemanticFrame(intent="blocked")
    if normalized in {"你好", "您好", "嗨", "hello", "hi"}:
        return SemanticFrame(intent="greeting")
    if normalized in {"确认", "确认提交", "是的", "对", "好的"}:
        return SemanticFrame(intent="confirm")
    if normalized in {"取消", "算了", "不提交", "不要了"}:
        return SemanticFrame(intent="cancel")
    if "转人工" in normalized or "人工客服" in normalized:
        return SemanticFrame(
            intent="handoff",
            slots={**_order_slots(normalized), "message": normalized},
            references=_references(normalized, state),
        )
    if any(term in normalized for term in ("退货", "换货", "维修", "售后")):
        issue_type = (
            "return"
            if "退货" in normalized
            else "exchange"
            if "换货" in normalized
            else "repair"
            if "维修" in normalized
            else "other"
        )
        return SemanticFrame(
            intent="after_sales",
            slots={
                **_order_slots(normalized),
                "issue_type": issue_type,
                "issue_description": normalized,
            },
            references=_references(normalized, state),
        )
    if "物流" in normalized or "快递" in normalized or "到哪" in normalized:
        return SemanticFrame(
            intent="logistics",
            slots=_order_slots(normalized),
            references=_references(normalized, state),
        )
    if "订单" in normalized:
        return SemanticFrame(
            intent="order",
            slots=_order_slots(normalized),
            references=_references(normalized, state),
        )

    references = _references(normalized, state)
    slots = _product_slots(normalized)
    continuation = any(term in normalized for term in _CONTINUATION_TERMS)
    manual = any(term in normalized for term in _MANUAL_TERMS)
    if manual:
        return SemanticFrame(
            intent="product_fact",
            slots=slots,
            references=references,
            question=normalized,
            requires_manual_evidence=True,
        )
    if any(term in normalized for term in _PRICE_TERMS):
        return SemanticFrame(
            intent="product_fact",
            slots={**slots, "fact_type": "catalog"},
            references=references,
            question=normalized,
        )
    if "对比" in normalized or "比较" in normalized or "区别" in normalized:
        return SemanticFrame(
            intent="compare_products",
            slots=slots,
            references=references,
        )
    if (
        "推荐" in normalized
        or "找" in normalized
        or "有没有" in normalized
        or continuation
        or _has_product_context(state)
        and slots
    ):
        return SemanticFrame(
            intent="recommend_products",
            slots=slots,
            references=references,
            continuation=continuation,
            requested_count=_requested_count(normalized),
        )
    return SemanticFrame(intent="other", slots=slots, references=references)


def _product_slots(query: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    remove_filters: list[str] = []
    for raw_category in ("鼠标", "键盘", "耳机", "音箱", "摄像头"):
        if raw_category not in query:
            continue
        if any(cue in query for cue in (f"不要{raw_category}", f"排除{raw_category}")):
            slots["excluded_category"] = normalize_product_category(raw_category) or raw_category
            continue
        normalized = normalize_product_category(raw_category)
        if normalized:
            slots["category"] = normalized
            slots["keyword"] = raw_category
    if "不限品牌" in query or "不要品牌" in query:
        remove_filters.append("brand")
    if "不限价格" in query:
        remove_filters.extend(["price_min", "price_max"])
    if remove_filters:
        slots["remove_filters"] = remove_filters
    return slots


def _references(
    query: str,
    state: CustomerServiceState,
) -> list[ReferenceExpression]:
    result: list[ReferenceExpression] = []
    for ordinal_match in _ORDINAL_RE.finditer(query):
        raw = ordinal_match.group(1)
        ordinal = int(raw) - 1 if raw.isdigit() else _ORDINALS[raw]
        category_hint = next(
            (value for value in ("鼠标", "键盘", "耳机", "订单") if value in query),
            None,
        )
        result.append(
            ReferenceExpression(
                text=ordinal_match.group(0),
                ordinal=ordinal,
                category_hint=category_hint,
            )
        )
    if result:
        return result
    for batch in reversed(state.candidate_batches):
        for item in batch.items:
            if item.product_code.casefold() in query.casefold():
                result.append(
                    ReferenceExpression(
                        text=item.product_code,
                        explicit_code=item.product_code,
                    )
                )
            if item.name.casefold() in query.casefold():
                result.append(
                    ReferenceExpression(
                        text=item.name,
                        explicit_name=item.name,
                    )
                )
        if result:
            return result
    explicit_tokens = re.findall(
        r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{2,64}\b",
        query,
    )
    if explicit_tokens:
        return [
            ReferenceExpression(
                text=token,
                explicit_code=token,
            )
            for token in dict.fromkeys(explicit_tokens)
        ]
    return result


def _order_slots(query: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    phone_match = re.search(r"(?:手机|手机号|尾号|后四位)\D{0,6}(\d{4})", query)
    if phone_match:
        slots["phone_last4"] = phone_match.group(1)
    explicit_order = re.search(
        r"(?:订单号?|订单)\D{0,4}([A-Za-z0-9][A-Za-z0-9_-]{3,63})",
        query,
    )
    if explicit_order and explicit_order.group(1) != slots.get("phone_last4"):
        slots["order_ref"] = explicit_order.group(1)
    elif match := re.search(r"\b\d{8,64}\b", query):
        slots["order_ref"] = match.group(0)
    return slots


def _requested_count(query: str) -> int | None:
    match = re.search(r"([1-5一二三四五])\s*(?:个|款|件|只)", query)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else _ORDINALS[raw] + 1


def _has_product_context(state: CustomerServiceState) -> bool:
    return bool(state.candidate_batches or state.filters)


async def _llm_understand(
    query: str,
    state: CustomerServiceState,
    messages: list[dict[str, Any]],
) -> SemanticFrame:
    schema = SemanticFrame.model_json_schema()
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "你只提取用户表达的语义，不选择工具，不声明数据库实体。"
                    "输出 SemanticFrame；无法确定时 intent=other。"
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "query": query,
                        "current_filters": state.filters,
                        "recent_messages": messages[-6:],
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "emit_semantic_frame",
                    "description": "输出用户语义",
                    "parameters": schema,
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "emit_semantic_frame"},
        },
        temperature=0,
        enable_thinking=False,
    )
    try:
        response = await asyncio.to_thread(LLMFactory.get_llm().chat, request)
        if response.tool_calls:
            return SemanticFrame.model_validate(response.tool_calls[0].arguments)
    except Exception:
        pass
    return SemanticFrame(intent="other")


def _merge(rule: SemanticFrame, llm: SemanticFrame) -> SemanticFrame:
    if rule.intent != "other":
        return rule
    return llm.model_copy(
        update={
            "slots": {**rule.slots, **llm.slots},
            "references": rule.references or llm.references,
        }
    )
