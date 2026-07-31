from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    ReferenceExpression,
    SemanticFrame,
)
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from pydantic import BaseModel, ConfigDict, Field

ReadOnlyCapability = Literal[
    "search_products",
    "recommend_products",
    "compare_products",
    "query_order",
    "query_logistics",
    "product_manual_fact",
    "none",
]


class ReadOnlyToolSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: ReadOnlyCapability
    slots: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=500)


class ReadOnlyFallbackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triggered: bool = True
    rewritten_query: str
    suggestion: ReadOnlyToolSuggestion | None = None
    semantic_frame: SemanticFrame | None = None
    failure_reason: str | None = None


async def recommend_read_only_capability(
    *,
    rewritten_query: str,
    state: CustomerServiceState,
) -> ReadOnlyFallbackResult:
    request = _build_request(rewritten_query=rewritten_query, state=state)
    try:
        response = await asyncio.to_thread(LLMFactory.get_llm().chat, request)
        if not response.tool_calls:
            return ReadOnlyFallbackResult(
                rewritten_query=rewritten_query,
                failure_reason="llm_returned_no_tool_call",
            )
        suggestion = ReadOnlyToolSuggestion.model_validate(
            response.tool_calls[0].arguments
        )
        frame = _to_semantic_frame(suggestion, rewritten_query)
        return ReadOnlyFallbackResult(
            rewritten_query=rewritten_query,
            suggestion=suggestion,
            semantic_frame=frame,
            failure_reason=(
                "no_read_only_capability" if frame is None else None
            ),
        )
    except Exception as exc:
        return ReadOnlyFallbackResult(
            rewritten_query=rewritten_query,
            failure_reason=f"{type(exc).__name__}:{exc}",
        )


def _build_request(
    *,
    rewritten_query: str,
    state: CustomerServiceState,
) -> LLMRequest:
    schema = ReadOnlyToolSuggestion.model_json_schema()
    active_batch = state.product.active_batch
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "你是智能客服只读能力推荐器。当前本地状态机无法承接用户请求。"
                    "只能从给定白名单中推荐一个只读能力，并提取用户明确表达的候选槽位。"
                    "禁止推荐写操作，禁止虚构商品、订单、说明书或数据库事实，"
                    "禁止把历史候选当作用户明确指定的实体。"
                    "无法安全匹配时 capability 必须为 none。"
                    "你只提供建议，Python 状态机、Resolver 和 Adapter 将独立校验。"
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "rewritten_query": rewritten_query,
                        "read_only_capabilities": {
                            "search_products": "按商品名称、类型、型号等条件查询商品",
                            "recommend_products": "按用户需求推荐商品",
                            "compare_products": "对比用户明确指定的多个商品",
                            "query_order": "查询订单或列出用户订单",
                            "query_logistics": "查询用户明确指向订单的物流",
                            "product_manual_fact": (
                                "查询商品使用、连接、兼容、充电、按键等说明书事实；"
                                "执行时必须先由商品工具验证商品"
                            ),
                            "none": "没有合适且安全的只读能力",
                        },
                        "trusted_state_summary": {
                            "active_product_category": state.product.active_category,
                            "active_product_codes": (
                                [item.product_code for item in active_batch.items]
                                if active_batch is not None
                                else []
                            ),
                            "active_order_ref": state.active_order_ref,
                            "order_candidate_count": len(state.order_candidates),
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "recommend_read_only_capability",
                    "description": "推荐一个只读客服能力及用户明确表达的候选槽位",
                    "parameters": schema,
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "recommend_read_only_capability"},
        },
        temperature=0,
        enable_thinking=False,
        metadata={"purpose": "customer_service_read_only_tool_fallback"},
    )


def _to_semantic_frame(
    suggestion: ReadOnlyToolSuggestion,
    rewritten_query: str,
) -> SemanticFrame | None:
    if suggestion.capability == "none":
        return None
    slots = _safe_slots(suggestion.slots)
    if suggestion.capability in {"search_products", "recommend_products"}:
        return SemanticFrame(
            intent=suggestion.capability,
            slots=slots,
            requested_count=_safe_requested_count(slots.pop("requested_count", None)),
        )
    if suggestion.capability == "compare_products":
        raw_codes = slots.pop("product_codes", [])
        references = (
            [
                ReferenceExpression(text=code, explicit_code=code)
                for code in raw_codes
                if isinstance(code, str) and code.strip()
            ]
            if isinstance(raw_codes, list)
            else []
        )
        return SemanticFrame(
            intent="compare_products",
            slots=slots,
            references=references,
        )
    if suggestion.capability == "query_order":
        return SemanticFrame(intent="order", slots=slots)
    if suggestion.capability == "query_logistics":
        return SemanticFrame(intent="logistics", slots=slots)
    if suggestion.capability == "product_manual_fact":
        product_code = slots.get("product_code")
        keyword = slots.get("keyword")
        references = []
        if isinstance(product_code, str):
            references.append(
                ReferenceExpression(text=product_code, explicit_code=product_code)
            )
        elif isinstance(keyword, str):
            references.append(
                ReferenceExpression(text=keyword, explicit_name=keyword)
            )
        return SemanticFrame(
            intent="product_fact",
            slots=slots,
            references=references,
            question=rewritten_query,
            requires_manual_evidence=True,
        )
    return None


def _safe_slots(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "keyword",
        "category",
        "brand",
        "model",
        "product_code",
        "product_codes",
        "order_ref",
        "question_predicate",
        "requested_count",
    }
    return {
        key: value
        for key, value in raw.items()
        if key in allowed and value not in (None, "", [], {})
    }


def _safe_requested_count(value: Any) -> int | None:
    return value if isinstance(value, int) and 1 <= value <= 5 else None
