"""Layer 1b: LLM 语义提取。

仅在 rules.py 返回 None 时调用。
包含共享 schema（IntentEnum / UnderstandingOutput）和 LLM 调用逻辑。
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

from backend.app.agents.customer_service_v2.session import SessionState
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 共享 Schema
# ---------------------------------------------------------------------------


class IntentEnum(StrEnum):
    """V2 意图枚举。"""

    RECOMMEND = "recommend"
    SEARCH = "search"
    COMPARE = "compare"
    FACT = "fact"
    ORDER = "order"
    LOGISTICS = "logistics"
    AFTER_SALES = "after_sales"
    HANDOFF = "handoff"
    GREETING = "greeting"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    BLOCKED = "blocked"
    OTHER = "other"


class UnderstandingOutput(BaseModel):
    """理解层统一输出。规则层和 LLM 层都输出这个结构。

    LLM 只做语义提取，不做流程判断。
    """

    intent: IntentEnum
    category: str | None = None
    keyword: str | None = None
    brand: str | None = None
    model: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    use_cases: list[str] = Field(default_factory=list, max_length=5)
    features: list[str] = Field(default_factory=list, max_length=5)
    target_refs: list[str] = Field(default_factory=list, max_length=5)
    order_ref: str | None = None
    phone_last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    after_sales_reason: str | None = None
    recommendation_count: int | None = Field(default=None, ge=1, le=5)
    filter_operations: dict[str, str] = Field(default_factory=dict)

    @field_validator("filter_operations")
    @classmethod
    def validate_filter_ops(cls, v: dict[str, str]) -> dict[str, str]:
        allowed_ops = {"SET", "REMOVE"}
        return {k: op for k, op in v.items() if op in allowed_ops}


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_understanding",
        "description": "从用户输入中提取结构化意图和实体信息",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [e.value for e in IntentEnum],
                    "description": "用户意图",
                },
                "category": {"type": "string", "description": "商品品类，如鼠标、键盘"},
                "keyword": {"type": "string", "description": "搜索关键词"},
                "brand": {"type": "string", "description": "品牌"},
                "model": {"type": "string", "description": "型号"},
                "price_min": {"type": "number", "description": "最低价格"},
                "price_max": {"type": "number", "description": "最高价格"},
                "use_cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "使用场景，如办公、游戏、出差",
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "产品特性，如静音、无线、机械轴",
                },
                "target_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用户引用的目标（原文），如第一个、G304",
                },
                "order_ref": {"type": "string", "description": "订单号"},
                "phone_last4": {"type": "string", "description": "手机尾号4位"},
                "after_sales_reason": {"type": "string", "description": "售后原因"},
                "recommendation_count": {
                    "type": "integer",
                    "description": "用户要求的推荐数量",
                },
                "filter_operations": {
                    "type": "object",
                    "description": "筛选条件修改操作，key为字段名，value为SET或REMOVE",
                },
            },
            "required": ["intent"],
        },
    },
}


def _build_context_message(session: SessionState) -> str:
    """构建上下文信息（active_products + last_filters）。"""
    parts: list[str] = []
    if session.active_products:
        items = ", ".join(
            f"{p.ref}.{p.name}({p.product_code})" for p in session.active_products
        )
        parts.append(f"当前候选商品: [{items}]")
    if session.last_filters:
        parts.append(f"上轮筛选条件: {json.dumps(session.last_filters, ensure_ascii=False)}")
    if session.active_order_ref:
        parts.append(f"当前订单: {session.active_order_ref}")
    return "\n".join(parts)


def _build_history_context(messages: list[dict[str, Any]]) -> str:
    """从历史消息中提取最近 3 轮精简上下文。"""
    if not messages:
        return ""
    recent = messages[-6:]  # 最多 3 轮 (user+assistant)*3
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:200]
        if role in ("user", "assistant") and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def llm_understand(
    query: str,
    session: SessionState,
    messages: list[dict[str, Any]],
) -> UnderstandingOutput:
    """调用 LLM 进行结构化语义提取。失败时降级到 intent=other 触发追问。"""
    from backend.app.agents.customer_service_v2.prompts import UNDERSTANDING_SYSTEM_PROMPT

    context = _build_context_message(session)
    history = _build_history_context(messages)

    user_content_parts = [f"用户输入: {query}"]
    if context:
        user_content_parts.append(f"对话状态:\n{context}")
    if history:
        user_content_parts.append(f"近期对话:\n{history}")

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=UNDERSTANDING_SYSTEM_PROMPT),
            LLMMessage(role="user", content="\n\n".join(user_content_parts)),
        ],
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "extract_understanding"}},
        temperature=0.0,
        enable_thinking=False,
    )

    try:
        llm = LLMFactory.get_llm()
        response = await _call_llm(llm, request)
        return _parse_response(response)
    except Exception:
        logger.warning("V2 understanding LLM call failed, fallback to other", exc_info=True)
        return UnderstandingOutput(intent=IntentEnum.OTHER)


async def _call_llm(llm: Any, request: LLMRequest) -> Any:
    """兼容同步/异步 LLM 调用。"""
    import asyncio

    if hasattr(llm, "achat"):
        return await llm.achat(request)
    return await asyncio.to_thread(llm.chat, request)


def _parse_response(response: Any) -> UnderstandingOutput:
    """从 LLM 响应中解析 tool_call 参数为 UnderstandingOutput。"""
    if response.tool_calls:
        args = response.tool_calls[0].arguments
        return UnderstandingOutput.model_validate(args)
    # 没有 tool_call，尝试从 answer 中解析 JSON
    answer = response.answer.strip()
    if answer.startswith("{"):
        data = json.loads(answer)
        return UnderstandingOutput.model_validate(data)
    # 完全无法解析
    return UnderstandingOutput(intent=IntentEnum.OTHER)
