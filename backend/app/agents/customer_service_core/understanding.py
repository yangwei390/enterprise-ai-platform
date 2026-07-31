from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    DialogueContextMessage,
    ReferenceExpression,
    SemanticFrame,
    UnderstandingContext,
)
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.schemas.product import normalize_product_category
from pydantic import BaseModel, ConfigDict, Field


class UnderstandingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: SemanticFrame
    rule_frame: SemanticFrame
    rewritten_query: str | None = None
    rewritten_rule_frame: SemanticFrame | None = None
    llm_used: bool = False
    llm_failure_reason: str | None = None
    llm_context: UnderstandingContext | None = None
    merge: dict[str, Any] = Field(default_factory=dict)


class QueryRewriteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(min_length=1, max_length=2000)


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
    "还有么",
    "其他的还有",
    "别的还有",
    "换一个",
    "换一款",
    "再来一个",
    "再推荐",
    "不要这个",
)
_ORDINAL_RE = re.compile(r"第\s*([一二三四五1-5])\s*(?:个|款|件|只|笔)?")
_ORDINALS = {"一": 0, "二": 1, "两": 1, "三": 2, "四": 3, "五": 4}


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
    context = _build_understanding_context(query, state, messages)
    rewritten_query, failure_reason = await _llm_rewrite(context)
    rewritten_rule_frame = _rule_understand(rewritten_query, state)
    return UnderstandingResult(
        frame=rewritten_rule_frame,
        rule_frame=rule_frame,
        rewritten_query=rewritten_query,
        rewritten_rule_frame=rewritten_rule_frame,
        llm_used=True,
        llm_failure_reason=failure_reason,
        llm_context=context,
        merge={"source": "llm_query_rewrite_then_rules", "conflicts": []},
    )


def _rule_understand(query: str, state: CustomerServiceState) -> SemanticFrame:
    normalized = query.strip()
    lowered = normalized.casefold()
    if any(term in lowered for term in _INJECTION_TERMS):
        return SemanticFrame(intent="blocked")
    if normalized in {"你好", "您好", "嗨", "hello", "hi"}:
        return SemanticFrame(intent="greeting")
    if (
        (
            normalized in {"确认", "是的", "对", "好的", "可以", "需要", "要", "查吧"}
            or normalized in {"帮我查", "帮我查一下", "查一下"}
        )
        and state.product.pending_query is not None
        and state.pending_after_sales is None
    ):
        return SemanticFrame(
            intent="confirm_pending_product_query",
            slots={"confirmation": True},
        )
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
    if (
        references
        and _is_ordinal_only_followup(normalized, references)
        and state.product.last_question is not None
        and state.product.active_batch is not None
        and state.product.last_question.batch_id == state.product.active_batch.batch_id
    ):
        predicate = state.product.last_question.predicate
        return SemanticFrame(
            intent="product_fact",
            slots={"question_predicate": predicate},
            references=references,
            question=_question_for_predicate(predicate),
            requires_manual_evidence=predicate != "price",
        )
    if manual:
        if _is_different_product_category(slots, state):
            return SemanticFrame(
                intent="product_fact_with_selection",
                slots={
                    **slots,
                    "question_predicate": _question_predicate(normalized),
                },
                question=normalized,
                requested_count=1,
                requires_manual_evidence=True,
            )
        return SemanticFrame(
            intent="product_fact",
            slots={**slots, "question_predicate": _question_predicate(normalized)},
            references=references,
            question=normalized,
            requires_manual_evidence=True,
        )
    if any(term in normalized for term in _PRICE_TERMS):
        if _is_different_product_category(slots, state):
            return SemanticFrame(
                intent="product_fact_with_selection",
                slots={**slots, "question_predicate": "price"},
                question=normalized,
                requested_count=1,
            )
        return SemanticFrame(
            intent="product_fact",
            slots={
                **slots,
                "fact_type": "catalog",
                "question_predicate": "price",
            },
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
        or "查询" in normalized
        or "有没有" in normalized
        or _is_product_availability_question(normalized, slots)
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
    if "keyword" not in slots:
        keyword = _open_product_keyword(query)
        if keyword is not None:
            slots["keyword"] = keyword
    return slots


def _open_product_keyword(query: str) -> str | None:
    action = re.search(r"(?:推荐|找|查询)", query)
    if action is None:
        return None
    remainder = query[action.end() :].strip()
    remainder = re.sub(r"^(?:给我|一下)", "", remainder).strip()
    remainder = re.sub(
        r"^(?:(?:一|二|两|三|四|五|[1-5])?(?:个|款|件|只)|一些)",
        "",
        remainder,
    ).strip()
    remainder = re.sub(r"[，,。.!！?？\s]+$", "", remainder)
    remainder = re.sub(r"[吗么嘛呢]$", "", remainder).strip()
    if not remainder or remainder in {"商品", "产品", "东西", "其他", "别的"}:
        return None
    return remainder[:128]


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
    batch = state.product.active_batch
    if batch is not None:
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
    match = re.search(r"([1-5一二两三四五])\s*(?:个|款|件|只)", query)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else _ORDINALS[raw] + 1


def _has_product_context(state: CustomerServiceState) -> bool:
    return bool(state.product.active_batch or state.product.filters)


async def _llm_rewrite(
    context: UnderstandingContext,
) -> tuple[str, str | None]:
    schema = QueryRewriteOutput.model_json_schema()
    dialogue_messages = [
        LLMMessage(role=message.role, content=message.content)
        for message in context.recent_dialogue
    ]
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "你是一个查询重写专家。请根据提供的历史对话上下文，将用户的"
                    "最新回复重写为一个独立、完整、无歧义的句子。\n"
                    "规则：\n"
                    "1. 如果用户使用了代词（他、那个、上一个），请替换为具体的实体名词。\n"
                    "2. 如果用户是在回答上文的问题（例如提供单号），请把问题和答案合并"
                    "（例：“123” -> “我的单号是123”）。\n"
                    "3. 如果用户的最新回复本身已经很完整，且与上文无关，原样输出，"
                    "不要强行拼接。\n"
                    "4. 仅输出重写后的句子，不要任何解释。"
                ),
            ),
            *dialogue_messages,
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "task": "仅重写当前用户问题",
                        "query": context.raw_query,
                        "dialog_focus": {
                            "pending_product_query": (
                                context.pending_product_query.model_dump(mode="json")
                                if context.pending_product_query is not None
                                else None
                            )
                        },
                        "business_state_summary": {
                            "active_product_category": context.active_product_category,
                            "active_product_codes": context.active_product_codes,
                        },
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
                    "name": "emit_rewritten_query",
                    "description": "仅输出重写后的完整问题",
                    "parameters": schema,
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "emit_rewritten_query"},
        },
        temperature=0,
        enable_thinking=False,
    )
    try:
        response = await asyncio.to_thread(LLMFactory.get_llm().chat, request)
        if response.tool_calls:
            output = QueryRewriteOutput.model_validate(response.tool_calls[0].arguments)
            return output.rewritten_query.strip(), None
        return context.raw_query, "llm_returned_no_tool_call"
    except Exception as exc:
        return context.raw_query, f"{type(exc).__name__}:{exc}"


def _build_understanding_context(
    query: str,
    state: CustomerServiceState,
    messages: list[dict[str, Any]],
) -> UnderstandingContext:
    normalized_messages: list[DialogueContextMessage] = []
    for raw in messages:
        role = raw.get("role")
        content = raw.get("content")
        if role in {"user", "human"}:
            normalized_role = "user"
        elif role in {"assistant", "ai"}:
            normalized_role = "assistant"
        else:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized_messages.append(
            DialogueContextMessage(
                role=normalized_role,
                content=content.strip()[:2000],
            )
        )
    if (
        normalized_messages
        and normalized_messages[-1].role == "user"
        and normalized_messages[-1].content == query.strip()
    ):
        normalized_messages.pop()

    complete_turns: list[list[DialogueContextMessage]] = []
    pending_user: DialogueContextMessage | None = None
    pending_assistant: DialogueContextMessage | None = None
    for message in normalized_messages:
        if message.role == "user":
            if pending_user is not None and pending_assistant is not None:
                complete_turns.append([pending_user, pending_assistant])
            pending_user = message
            pending_assistant = None
        elif pending_user is not None:
            pending_assistant = message
    if pending_user is not None and pending_assistant is not None:
        complete_turns.append([pending_user, pending_assistant])

    recent_dialogue = [message for turn in complete_turns[-3:] for message in turn]
    batch = state.product.active_batch
    return UnderstandingContext(
        raw_query=query,
        recent_dialogue=recent_dialogue,
        active_product_category=state.product.active_category,
        active_product_codes=(
            [item.product_code for item in batch.items] if batch is not None else []
        ),
        pending_product_query=state.product.pending_query,
    )


def _is_different_product_category(
    slots: dict[str, Any],
    state: CustomerServiceState,
) -> bool:
    category = slots.get("category")
    active = state.product.active_category
    return isinstance(category, str) and active is not None and category != active


def _is_product_availability_question(
    query: str,
    slots: dict[str, Any],
) -> bool:
    return bool(slots) and re.fullmatch(r"有.+[吗么嘛?？]", query) is not None


def _is_ordinal_only_followup(
    query: str,
    references: list[ReferenceExpression],
) -> bool:
    stripped = _ORDINAL_RE.sub("", query)
    stripped = re.sub(r"[\s，,。.!！?？呢吗呀]+", "", stripped)
    return bool(references) and not stripped


def _question_predicate(query: str) -> str:
    if "蓝牙" in query:
        return "bluetooth_connectivity"
    if "充电" in query:
        return "charging"
    if "兼容" in query:
        return "compatibility"
    if "按键" in query:
        return "buttons"
    return "features"


def _question_for_predicate(predicate: str) -> str:
    return {
        "bluetooth_connectivity": "该商品支持蓝牙吗",
        "charging": "该商品支持充电吗",
        "compatibility": "该商品兼容吗",
        "price": "该商品多少钱",
        "features": "该商品有什么特点",
        "buttons": "该商品有哪些按键",
    }.get(predicate, "该商品有什么特点")
