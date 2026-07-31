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
from backend.app.tools.registry import get_tool_registry
from pydantic import BaseModel, ConfigDict, Field

ReadOnlyToolName = Literal[
    "search_products",
    "recommend_products",
    "compare_products",
    "query_order",
    "query_logistics",
    "none",
]

_READ_ONLY_TOOL_NAMES = (
    "search_products",
    "recommend_products",
    "compare_products",
    "query_order",
    "query_logistics",
)
_PROTECTED_ARGUMENTS = {"knowledge_base_id", "excluded_product_codes"}


class ReadOnlyToolSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: ReadOnlyToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
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
        validated_arguments = _validate_suggested_arguments(suggestion)
        frame = _to_semantic_frame(
            suggestion,
            validated_arguments,
        )
        return ReadOnlyFallbackResult(
            rewritten_query=rewritten_query,
            suggestion=suggestion,
            semantic_frame=frame,
            failure_reason=(
                "no_read_only_tool" if frame is None else None
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
    tool_catalog = _read_only_tool_catalog()
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "你是智能客服只读能力推荐器。当前本地状态机无法承接用户请求。"
                    "只能从给定白名单中推荐一个只读 Tool，并严格按该 Tool "
                    "的 parameters 生成 arguments。"
                    "禁止推荐写操作，禁止虚构商品、订单、说明书或数据库事实，"
                    "禁止把历史候选当作用户明确指定的实体。"
                    "参数名必须与 parameters 完全一致，不得自创 product_name 等字段。"
                    "knowledge_base_id 和 excluded_product_codes 是平台保护参数，禁止输出。"
                    "无法安全匹配时 tool_name 必须为 none。"
                    "你只提供建议，Python 状态机、Resolver 和 Adapter 将独立校验。"
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "rewritten_query": rewritten_query,
                        "read_only_tools": tool_catalog,
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
    arguments: dict[str, Any],
) -> SemanticFrame | None:
    if suggestion.tool_name == "none":
        return None
    slots = dict(arguments)
    if suggestion.tool_name in {"search_products", "recommend_products"}:
        return SemanticFrame(
            intent=suggestion.tool_name,
            slots=slots,
            requested_count=_safe_requested_count(slots.pop("page_size", None)),
        )
    if suggestion.tool_name == "compare_products":
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
    if suggestion.tool_name == "query_order":
        return SemanticFrame(intent="order", slots=slots)
    if suggestion.tool_name == "query_logistics":
        return SemanticFrame(intent="logistics", slots=slots)
    return None


def _validate_suggested_arguments(
    suggestion: ReadOnlyToolSuggestion,
) -> dict[str, Any]:
    if suggestion.tool_name == "none":
        if suggestion.arguments:
            raise ValueError("none tool must not contain arguments")
        return {}
    protected = _PROTECTED_ARGUMENTS & suggestion.arguments.keys()
    if protected:
        raise ValueError(f"protected tool arguments: {sorted(protected)}")
    tool = get_tool_registry().get_tool(suggestion.tool_name, require_enabled=True)
    if tool is None:
        raise ValueError(f"read-only tool not found: {suggestion.tool_name}")
    validated = tool.args_schema.model_validate(suggestion.arguments).model_dump()
    return {
        key: validated[key]
        for key in suggestion.arguments
        if key in validated
    }


def _read_only_tool_catalog() -> list[dict[str, Any]]:
    registry = get_tool_registry()
    catalog: list[dict[str, Any]] = []
    for tool_name in _READ_ONLY_TOOL_NAMES:
        tool = registry.get_tool(tool_name, require_enabled=True)
        if tool is None:
            continue
        parameters = tool.args_schema.model_json_schema()
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            for protected in _PROTECTED_ARGUMENTS:
                properties.pop(protected, None)
        required = parameters.get("required")
        if isinstance(required, list):
            parameters["required"] = [
                name for name in required if name not in _PROTECTED_ARGUMENTS
            ]
        catalog.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            }
        )
    catalog.append(
        {
            "name": "none",
            "description": "没有合适且安全的只读 Tool",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    )
    return catalog


def _safe_requested_count(value: Any) -> int | None:
    return value if isinstance(value, int) and 1 <= value <= 5 else None
