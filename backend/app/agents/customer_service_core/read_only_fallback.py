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
    "knowledge_search",
    "none",
]
ResponseMode = Literal[
    "product_list",
    "product_recommendation",
    "product_catalog_detail",
    "product_feature_match",
    "product_use_case_match",
    "product_comparison",
    "manual_fact",
    "order_list",
    "order_detail",
    "logistics_status",
    "clarification",
]

_READ_ONLY_TOOL_NAMES = (
    "search_products",
    "recommend_products",
    "compare_products",
    "query_order",
    "query_logistics",
)
_PROTECTED_ARGUMENTS = {"knowledge_base_id", "excluded_product_codes"}


class ReadOnlyQuestionSlots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_case: str | None = Field(default=None, min_length=1, max_length=64)
    feature: str | None = Field(default=None, min_length=1, max_length=64)
    manual_question: str | None = Field(default=None, min_length=1, max_length=500)


class ReadOnlyToolSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: ReadOnlyToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    response_mode: ResponseMode
    question_slots: ReadOnlyQuestionSlots = Field(default_factory=ReadOnlyQuestionSlots)
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
        validated_arguments = _validate_suggested_arguments(suggestion, state)
        frame = _to_semantic_frame(
            suggestion,
            validated_arguments,
            rewritten_query=rewritten_query,
            state=state,
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
                    "先区分用户是在找商品，还是询问当前商品的事实。"
                    "当前商品已经确定时必须使用可信 product_code，不得把问题中的"
                    "用途或能力词当作 keyword 重新搜索。"
                    "价格、库存、features、use_cases 属于商品目录事实；"
                    "蓝牙、连接、充电、兼容、按键和使用方法属于说明书事实。"
                    "只有用户明确要求推荐、换一个或其他选择时才使用 recommend_products。"
                    "arguments 只表示 Tool 查询条件；回答目标必须通过 response_mode "
                    "和 question_slots 表达。"
                    "必须从 rewritten_query 提取用户明确说出的商品名称、"
                    "类别、型号、用途或其他查询条件，并写入对应 arguments。"
                    "不得因为 JSON Schema 字段可选就遗漏用户已明确表达的条件。"
                    "选择 Tool 后必须逐项检查该 Tool 的 parameter_requirements，"
                    "满足 schema_required_parameters 和当前 response_mode 的"
                    "business_required_rules。"
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
                            "active_product": _active_product_summary(state),
                            "visible_product_batch": (
                                [
                                    {
                                        "position": item.position + 1,
                                        "product_code": item.product_code,
                                        "name": item.name,
                                        "category": item.category,
                                        "primary_manual_document_id": (
                                            item.primary_manual_document_id
                                        ),
                                        "source": "verified_tool_result",
                                    }
                                    for item in active_batch.items
                                ]
                                if active_batch is not None else []
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
    *,
    rewritten_query: str,
    state: CustomerServiceState,
) -> SemanticFrame | None:
    if suggestion.tool_name == "none":
        return None
    slots = {
        **arguments,
        "response_mode": suggestion.response_mode,
        **suggestion.question_slots.model_dump(exclude_none=True),
    }
    if suggestion.tool_name == "knowledge_search":
        active_code = state.product.active_product_code
        if active_code is None:
            return None
        return SemanticFrame(
            intent="product_fact",
            slots=slots,
            references=[ReferenceExpression(text=active_code, explicit_code=active_code)],
            question=suggestion.question_slots.manual_question or rewritten_query,
            requires_manual_evidence=True,
        )
    if suggestion.tool_name in {"search_products", "recommend_products"}:
        if suggestion.response_mode in {
            "product_catalog_detail",
            "product_feature_match",
            "product_use_case_match",
        }:
            product_code = arguments.get("product_code")
            if not isinstance(product_code, str):
                return None
            return SemanticFrame(
                intent="product_fact",
                slots=slots,
                references=[
                    ReferenceExpression(text=product_code, explicit_code=product_code)
                ],
                question=rewritten_query,
            )
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
    state: CustomerServiceState,
) -> dict[str, Any]:
    if suggestion.tool_name == "none":
        _validate_response_contract(suggestion)
        if suggestion.arguments:
            raise ValueError("none tool must not contain arguments")
        return {}
    if suggestion.tool_name == "knowledge_search":
        _validate_response_contract(suggestion)
        if suggestion.arguments:
            raise ValueError("knowledge_search arguments are injected by Python")
        active = _active_product_summary(state)
        if active is None or active.get("primary_manual_document_id") is None:
            raise ValueError("trusted active product manual is missing")
        return {}
    protected = _PROTECTED_ARGUMENTS & suggestion.arguments.keys()
    if protected:
        raise ValueError(f"protected tool arguments: {sorted(protected)}")
    tool = get_tool_registry().get_tool(suggestion.tool_name, require_enabled=True)
    if tool is None:
        raise ValueError(f"read-only tool not found: {suggestion.tool_name}")
    validated = tool.args_schema.model_validate(suggestion.arguments).model_dump()
    _validate_response_contract(suggestion)
    product_code = suggestion.arguments.get("product_code")
    if isinstance(product_code, str) and suggestion.response_mode in {
        "product_catalog_detail",
        "product_feature_match",
        "product_use_case_match",
    }:
        trusted_codes = {
            item.product_code
            for item in (
                state.product.active_batch.items
                if state.product.active_batch is not None
                else []
            )
        }
        if product_code not in trusted_codes:
            raise ValueError("suggested product_code is outside trusted visible batch")
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
                **_tool_guidance(tool_name),
                "parameter_requirements": _parameter_requirements(
                    tool_name,
                    parameters,
                ),
                "parameters": parameters,
            }
        )
    knowledge_tool = registry.get_tool("knowledge_search", require_enabled=True)
    if knowledge_tool is not None:
        catalog.append(
            {
                "name": "knowledge_search",
                "description": "查询当前可信商品绑定的主说明书，回答使用和能力事实。",
                **_tool_guidance("knowledge_search"),
                "parameter_requirements": {
                    "schema_required_parameters": [],
                    "optional_parameters": [],
                    "business_required_rules": {
                        "manual_fact": [
                            "arguments 必须为空对象",
                            "question_slots.manual_question 必填",
                            "必须存在绑定主说明书的可信当前商品",
                        ]
                    },
                },
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                    "description": (
                        "LLM 不填写参数；Python 从可信商品状态注入 document_id、"
                        "knowledge_base_id 和完整问题。"
                    ),
                },
            }
        )
    catalog.append(
        {
            "name": "none",
            "description": "没有合适且安全的只读 Tool",
            "parameter_requirements": {
                "schema_required_parameters": [],
                "optional_parameters": [],
                "business_required_rules": {
                    "clarification": ["arguments 必须为空对象"]
                },
            },
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    )
    return catalog


def _active_product_summary(state: CustomerServiceState) -> dict[str, Any] | None:
    code = state.product.active_product_code
    batch = state.product.active_batch
    if code is None or batch is None:
        return None
    item = next((candidate for candidate in batch.items if candidate.product_code == code), None)
    if item is None:
        return None
    return {
        "product_code": item.product_code,
        "name": item.name,
        "category": item.category,
        "primary_manual_document_id": item.primary_manual_document_id,
        "source": "verified_tool_result",
    }


def _validate_response_contract(suggestion: ReadOnlyToolSuggestion) -> None:
    allowed = {
        "search_products": {
            "product_list",
            "product_catalog_detail",
            "product_feature_match",
            "product_use_case_match",
        },
        "recommend_products": {"product_recommendation"},
        "compare_products": {"product_comparison"},
        "query_order": {"order_list", "order_detail"},
        "query_logistics": {"logistics_status"},
        "knowledge_search": {"manual_fact"},
        "none": {"clarification"},
    }
    if suggestion.response_mode not in allowed[suggestion.tool_name]:
        raise ValueError("response_mode is incompatible with selected tool")
    if suggestion.tool_name == "search_products":
        if suggestion.response_mode == "product_list" and not _has_product_condition(
            suggestion.arguments
        ):
            raise ValueError("product_list requires at least one product query condition")
        if suggestion.response_mode in {
            "product_catalog_detail",
            "product_feature_match",
            "product_use_case_match",
        } and not isinstance(suggestion.arguments.get("product_code"), str):
            raise ValueError(f"{suggestion.response_mode} requires product_code")
    if suggestion.tool_name == "recommend_products" and not _has_product_condition(
        suggestion.arguments
    ):
        raise ValueError("product_recommendation requires at least one product query condition")
    if suggestion.response_mode == "order_list" and "order_ref" in suggestion.arguments:
        raise ValueError("order_list must not contain order_ref")
    if suggestion.response_mode == "order_detail" and not isinstance(
        suggestion.arguments.get("order_ref"), str
    ):
        raise ValueError("order_detail requires order_ref")
    if (
        suggestion.response_mode == "product_use_case_match"
        and suggestion.question_slots.use_case is None
    ):
        raise ValueError("product_use_case_match requires use_case")
    if (
        suggestion.response_mode == "product_feature_match"
        and suggestion.question_slots.feature is None
    ):
        raise ValueError("product_feature_match requires feature")
    if (
        suggestion.response_mode == "manual_fact"
        and suggestion.question_slots.manual_question is None
    ):
        raise ValueError("manual_fact requires manual_question")


def _tool_guidance(tool_name: str) -> dict[str, Any]:
    guidance: dict[str, dict[str, Any]] = {
        "search_products": {
            "purpose": "查询指定商品事实，或按明确条件检索商品目录。",
            "use_when": [
                "查询已确定商品的价格、库存、features 或 use_cases",
                "用户询问店内是否存在某类商品",
                "按明确名称、型号、类别或筛选条件查询商品",
            ],
            "do_not_use_when": [
                "用户明确要求推荐、换一个或其他选择",
                "用户询问说明书中的连接、充电、兼容、按键或使用方法",
                "当前商品已确定时，不得把用途或能力词作为 keyword 重新搜索",
            ],
            "argument_guidance": {
                "product_code": "查询当前已确定商品时必须使用可信商品编码",
                "keyword": "仅用于名称、型号或自由检索词，不得填用户询问的动作或用途",
                "required_use_cases": "仅用于筛选满足某用途的商品，不用于询问当前商品",
            },
            "returns": ["商品目录字段", "features", "use_cases", "主说明书ID"],
        },
        "recommend_products": {
            "purpose": "按用户明确需求推荐商品，并支持换一个或继续推荐。",
            "use_when": ["用户明确说推荐、换一个、再推荐或还有其他选择"],
            "do_not_use_when": ["用户仅询问当前商品的价格、能力或适用场景"],
            "argument_guidance": {
                "required_use_cases": "用户要求推荐适合某用途的商品时使用",
                "page_size": "用户未明确数量时使用 1",
            },
            "returns": ["商品", "推荐评分", "推荐原因", "无结果原因"],
        },
        "compare_products": {
            "purpose": "比较用户明确指定的两个及以上可信商品。",
            "use_when": ["用户明确要求比较或询问区别"],
            "do_not_use_when": ["商品不足两个或尚未明确"],
            "argument_guidance": {"product_codes": "只能使用可信可见商品编码"},
            "returns": ["逐字段对比", "缺失商品编码"],
        },
        "query_order": {
            "purpose": "查询当前用户订单列表或指定订单详情。",
            "use_when": ["用户查询订单列表或订单详情"],
            "do_not_use_when": ["用户只查询已确定订单的物流"],
            "argument_guidance": {"order_ref": "仅使用用户明确提供或可信状态中的订单号"},
            "returns": ["脱敏订单列表或订单详情"],
        },
        "query_logistics": {
            "purpose": "查询已确定订单的物流状态。",
            "use_when": ["用户查询某笔可信订单到哪了或物流进度"],
            "do_not_use_when": ["订单尚未明确"],
            "argument_guidance": {"order_ref": "必须是可信订单号"},
            "returns": ["物流状态和轨迹"],
        },
        "knowledge_search": {
            "purpose": "查询当前商品绑定说明书中的使用、连接和能力事实。",
            "use_when": ["蓝牙、连接、充电、兼容、按键、操作或使用方法"],
            "do_not_use_when": ["价格、库存、features、use_cases 等商品目录事实"],
            "argument_guidance": {
                "arguments": "必须为空；可信文档和知识库范围由 Python 注入"
            },
            "returns": ["有来源和引用的说明书答案"],
        },
    }
    return guidance[tool_name]


def _parameter_requirements(
    tool_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    properties = parameters.get("properties")
    property_names = list(properties) if isinstance(properties, dict) else []
    schema_required = parameters.get("required")
    required_names = (
        [name for name in schema_required if isinstance(name, str)]
        if isinstance(schema_required, list)
        else []
    )
    business_rules: dict[str, list[str]] = {
        "search_products": {
            "product_list": [
                "product_code、keyword、brand、category、model、价格、"
                "特征或用途条件中至少一项必填"
            ],
            "product_catalog_detail": ["product_code 必填"],
            "product_feature_match": [
                "product_code 必填",
                "question_slots.feature 必填",
            ],
            "product_use_case_match": [
                "product_code 必填",
                "question_slots.use_case 必填",
            ],
        },
        "recommend_products": {
            "product_recommendation": [
                "product_code、keyword、brand、category、model、价格、"
                "特征或用途条件中至少一项必填",
                "page_size 可选，用户未明确数量时为 1",
            ]
        },
        "compare_products": {
            "product_comparison": ["product_codes 必填，且必须包含 2 至 5 个不同编码"]
        },
        "query_order": {
            "order_list": ["order_ref 禁止填写"],
            "order_detail": ["order_ref 必填"],
        },
        "query_logistics": {
            "logistics_status": ["order_ref 必填"]
        },
    }[tool_name]
    return {
        "schema_required_parameters": required_names,
        "optional_parameters": [
            name for name in property_names if name not in required_names
        ],
        "business_required_rules": business_rules,
    }


def _has_product_condition(arguments: dict[str, Any]) -> bool:
    scalar_fields = {
        "product_code",
        "keyword",
        "brand",
        "category",
        "model",
        "price_min",
        "price_max",
    }
    list_fields = {
        "required_features",
        "excluded_features",
        "preferred_features",
        "required_use_cases",
        "preferred_use_cases",
        "features",
        "use_cases",
    }
    return any(arguments.get(field) not in {None, ""} for field in scalar_fields) or any(
        isinstance(arguments.get(field), list) and bool(arguments[field])
        for field in list_fields
    )


def _safe_requested_count(value: Any) -> int | None:
    return value if isinstance(value, int) and 1 <= value <= 5 else None
