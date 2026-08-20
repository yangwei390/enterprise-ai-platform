from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any, Literal

from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceState,
    ReferenceExpression,
    SemanticFrame,
)
from backend.app.agents.customer_service_core.stage_modes import (
    StageExecutionDetail,
    StageMode,
    intent_routing_mode,
    reference_interpretation_mode,
    slot_extraction_mode,
    tool_selection_mode,
)
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.llms.config import get_customer_service_reasoning_llm_config
from backend.app.tools.registry import get_tool_registry
from pydantic import BaseModel, ConfigDict, Field

CapabilityToolName = Literal[
    "search_products",
    "recommend_products",
    "compare_products",
    "query_order",
    "query_logistics",
    "knowledge_search",
    "create_after_sales_ticket",
    "create_human_handoff",
    "none",
]
ResponseMode = Literal[
    "product_list",
    "product_recommendation",
    "product_catalog_detail",
    "product_feature_match",
    "product_comparison",
    "manual_fact",
    "manual_fact_with_selection",
    "order_list",
    "order_detail",
    "logistics_status",
    "after_sales_draft",
    "human_handoff",
    "clarification",
]

_PROTECTED_ARGUMENTS = {"knowledge_base_id", "excluded_product_codes"}


class CapabilityQuestionSlots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str | None = Field(default=None, min_length=1, max_length=64)
    manual_question: str | None = Field(default=None, min_length=1, max_length=500)


class CapabilitySuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: CapabilityToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    response_mode: ResponseMode
    question_slots: CapabilityQuestionSlots = Field(default_factory=CapabilityQuestionSlots)
    reason: str = Field(default="", max_length=500)


class SearchProductListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    keyword: str = Field(
        min_length=1,
        max_length=128,
        description="必填。用户明确表达的商品名称、类别、型号或检索词。",
    )
    category: str | None = Field(default=None, max_length=128, description="商品类别。")
    brand: str | None = Field(default=None, max_length=128, description="商品品牌。")
    model: str | None = Field(default=None, max_length=128, description="商品型号。")
    price_min: Decimal | None = Field(default=None, ge=0, description="最低价格。")
    price_max: Decimal | None = Field(default=None, ge=0, description="最高价格。")
    required_features: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="必须具备的商品目录特征。",
    )
    required_use_cases: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="必须适合的用途。",
    )
    page_size: int = Field(default=20, ge=1, le=20, description="返回数量。")


class RecommendProductArgs(SearchProductListArgs):
    page_size: int = Field(default=1, ge=1, le=5, description="推荐数量，未明确时为 1。")


class ProductCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_code: str = Field(
        min_length=1,
        max_length=64,
        description="必填。来自可信当前商品或可见候选批次的商品编码。",
    )


class ProductFeatureArgs(ProductCodeArgs):
    feature: str = Field(
        min_length=1,
        max_length=64,
        description="必填。用户要核验的商品目录特征。",
    )


class CompareProductArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_codes: list[str] = Field(
        min_length=2,
        max_length=5,
        description="必填。2至5个用户明确指定或可信可见的商品编码。",
    )


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderRefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_ref: str = Field(
        min_length=4,
        max_length=64,
        description="必填。用户明确提供或可信状态中的订单号。",
    )


class ProductManualArgs(ProductCodeArgs):
    manual_question: str = Field(
        min_length=1,
        max_length=500,
        description="必填。需要从该商品说明书中核验的完整问题。",
    )


class ProductManualSelectionArgs(SearchProductListArgs):
    page_size: int = Field(default=1, ge=1, le=1, description="固定先验证一个商品。")
    manual_question: str = Field(
        min_length=1,
        max_length=500,
        description="必填。商品验证成功后需要从说明书中核验的完整问题。",
    )


class AfterSalesDraftArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_ref: str | None = Field(default=None, min_length=4, max_length=64)
    phone_last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    issue_type: Literal["quality", "repair", "return", "exchange", "other"] = "other"
    issue_description: str | None = Field(default=None, min_length=1, max_length=1000)


class HumanHandoffArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_ref: str | None = Field(default=None, min_length=4, max_length=64)
    phone_last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    reason: Literal["customer_request", "complaint", "tool_unavailable", "other"] = (
        "customer_request"
    )
    message: str = Field(min_length=2, max_length=1000)


class CapabilitySelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triggered: bool = True
    rewritten_query: str
    suggestion: CapabilitySuggestion | None = None
    semantic_frame: SemanticFrame | None = None
    failure_reason: str | None = None
    llm_call_count: int = Field(default=1, ge=0)
    stage_details: dict[str, StageExecutionDetail] = Field(default_factory=dict)


async def select_capability(
    *,
    rewritten_query: str,
    state: CustomerServiceState,
) -> CapabilitySelectionResult:
    request = _build_request(rewritten_query=rewritten_query, state=state)
    try:
        response = await asyncio.to_thread(
            LLMFactory.get_llm(
                config=get_customer_service_reasoning_llm_config(),
            ).chat,
            request,
        )
        if not response.tool_calls:
            return CapabilitySelectionResult(
                rewritten_query=rewritten_query,
                failure_reason="llm_returned_no_tool_call",
                stage_details=_failed_shared_stage_details(
                    rewritten_query,
                    "llm_returned_no_tool_call",
                ),
            )
        suggestion = _suggestion_from_native_tool_call(
            tool_name=response.tool_calls[0].name,
            arguments=response.tool_calls[0].arguments,
        )
        validated_arguments = _validate_suggested_arguments(suggestion, state)
        frame = _to_semantic_frame(
            suggestion,
            validated_arguments,
            rewritten_query=rewritten_query,
            state=state,
        )
        return CapabilitySelectionResult(
            rewritten_query=rewritten_query,
            suggestion=suggestion,
            semantic_frame=frame,
            failure_reason=(
                "no_capability" if frame is None else None
            ),
            stage_details=_shared_stage_details(
                rewritten_query=rewritten_query,
                suggestion=suggestion,
                semantic_frame=frame,
            ),
        )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}:{exc}"
        return CapabilitySelectionResult(
            rewritten_query=rewritten_query,
            failure_reason=failure_reason,
            stage_details=_failed_shared_stage_details(
                rewritten_query,
                failure_reason,
            ),
        )


def _shared_stage_details(
    *,
    rewritten_query: str,
    suggestion: CapabilitySuggestion,
    semantic_frame: SemanticFrame | None,
) -> dict[str, StageExecutionDetail]:
    shared_call_id = "heavy_semantic_tool_call"
    frame = semantic_frame or SemanticFrame(intent="other")
    domain = _domain_for_intent(frame.intent)
    common_input = {"rewritten_query": rewritten_query}
    return {
        "intent_routing": StageExecutionDetail(
            mode=intent_routing_mode(),
            source="shared_llm",
            executed=True,
            input=common_input,
            output={"intent": frame.intent, "domain": domain},
            shared_call_id=shared_call_id,
        ),
        "slot_extraction": StageExecutionDetail(
            mode=slot_extraction_mode(),
            source="shared_llm",
            executed=True,
            input=common_input,
            output={
                "slots": frame.slots,
                "question": frame.question,
                "requested_count": frame.requested_count,
            },
            shared_call_id=shared_call_id,
        ),
        "reference_interpretation": StageExecutionDetail(
            mode=reference_interpretation_mode(),
            source="shared_llm",
            executed=True,
            input=common_input,
            output={
                "references": [
                    reference.model_dump(mode="json") for reference in frame.references
                ]
            },
            shared_call_id=shared_call_id,
        ),
        "capability_selection": StageExecutionDetail(
            mode=tool_selection_mode(),
            source="shared_llm",
            executed=True,
            input=common_input,
            output={
                "tool_name": suggestion.tool_name,
                "arguments": suggestion.arguments,
                "response_mode": suggestion.response_mode,
            },
            shared_call_id=shared_call_id,
        ),
    }


def _failed_shared_stage_details(
    rewritten_query: str,
    failure_reason: str,
) -> dict[str, StageExecutionDetail]:
    failed_stages: tuple[tuple[str, StageMode], ...] = (
        ("intent_routing", intent_routing_mode()),
        ("slot_extraction", slot_extraction_mode()),
        ("reference_interpretation", reference_interpretation_mode()),
        ("capability_selection", tool_selection_mode()),
    )
    return {
        stage_name: StageExecutionDetail(
            mode=mode,
            source="shared_llm",
            executed=True,
            input={"rewritten_query": rewritten_query},
            failure_reason=failure_reason,
            shared_call_id="heavy_semantic_tool_call",
        )
        for stage_name, mode in failed_stages
    }


def _domain_for_intent(intent: str) -> str:
    if intent in {"recommend_products", "search_products", "compare_products"}:
        return "product"
    if intent in {"product_fact", "product_fact_with_selection"}:
        return "manual"
    if intent == "logistics":
        return "logistics"
    if intent == "order":
        return "order"
    if intent == "after_sales":
        return "after_sales"
    if intent == "handoff":
        return "handoff"
    return "general"


def _build_request(
    *,
    rewritten_query: str,
    state: CustomerServiceState,
) -> LLMRequest:
    active_batch = state.product.active_batch
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "你是智能客服受控能力调度器。"
                    "必须从提供的原生 Function Tools 中选择一个，"
                    "严格按所选 Function 的 JSON Schema 生成参数。"
                    "先区分用户是在找商品，还是询问当前商品的事实。"
                    "当前商品已经确定时必须使用可信 product_code，不得把问题中的"
                    "用途或能力词当作 keyword 重新搜索。"
                    "价格、库存、features、use_cases 属于商品目录事实；"
                    "蓝牙、连接、充电、兼容、按键、使用方法、保修和是否适合某种用途"
                    "都属于说明书事实，必须使用 search_product_manual。"
                    "只有用户明确要求推荐、换一个或其他选择时才使用 recommend_products。"
                    "必须从 rewritten_query 提取用户明确说出的商品名称、"
                    "类别、型号、用途或其他查询条件，填入所选 Function 的必填字段。"
                    "售后只允许选择创建草稿，绝不能选择确认提交；"
                    "禁止虚构商品、订单、说明书或数据库事实，"
                    "禁止把历史候选当作用户明确指定的实体。"
                    "无法安全匹配时调用 clarify_request。"
                    "Python 状态机、Resolver 和 Adapter 将独立校验。"
                ),
            ),
            LLMMessage(
                role="user",
                content=json.dumps(
                    {
                        "rewritten_query": rewritten_query,
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
                            "active_order_ref": state.order.active_order_ref,
                            "visible_order_batch": (
                                [
                                    {
                                        "position": item.position + 1,
                                        "order_ref": item.order_ref,
                                        "display_label": item.display_label,
                                        "source": "verified_tool_result",
                                    }
                                    for item in state.order.active_batch.items
                                ]
                                if state.order.active_batch is not None else []
                            ),
                            "pending_clarification_user_expression": (
                                state.pending_clarification.model_dump(mode="json")
                                if state.pending_clarification is not None
                                else None
                            ),
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        tools=_native_capability_tools(),
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=0,
        enable_thinking=False,
        metadata={"purpose": "customer_service_capability_selection"},
    )


def _suggestion_from_native_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> CapabilitySuggestion:
    if tool_name == "search_products":
        parsed = SearchProductListArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="search_products",
            arguments=parsed.model_dump(exclude_none=True),
            response_mode="product_list",
        )
    if tool_name == "recommend_products":
        parsed = RecommendProductArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="recommend_products",
            arguments=parsed.model_dump(exclude_none=True),
            response_mode="product_recommendation",
        )
    if tool_name == "get_product_catalog_detail":
        parsed = ProductCodeArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="search_products",
            arguments=parsed.model_dump(),
            response_mode="product_catalog_detail",
        )
    if tool_name == "check_product_feature":
        parsed = ProductFeatureArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="search_products",
            arguments={"product_code": parsed.product_code},
            response_mode="product_feature_match",
            question_slots=CapabilityQuestionSlots(feature=parsed.feature),
        )
    if tool_name == "compare_products":
        parsed = CompareProductArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="compare_products",
            arguments=parsed.model_dump(),
            response_mode="product_comparison",
        )
    if tool_name == "list_orders":
        EmptyArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="query_order",
            response_mode="order_list",
        )
    if tool_name == "get_order_detail":
        parsed = OrderRefArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="query_order",
            arguments=parsed.model_dump(),
            response_mode="order_detail",
        )
    if tool_name == "query_logistics":
        parsed = OrderRefArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="query_logistics",
            arguments=parsed.model_dump(),
            response_mode="logistics_status",
        )
    if tool_name == "search_product_manual":
        parsed = ProductManualArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="knowledge_search",
            arguments={"product_code": parsed.product_code},
            response_mode="manual_fact",
            question_slots=CapabilityQuestionSlots(
                manual_question=parsed.manual_question,
            ),
        )
    if tool_name == "select_product_for_manual":
        parsed = ProductManualSelectionArgs.model_validate(arguments)
        query_arguments = parsed.model_dump(
            exclude={"manual_question"},
            exclude_none=True,
        )
        return CapabilitySuggestion(
            tool_name="recommend_products",
            arguments=query_arguments,
            response_mode="manual_fact_with_selection",
            question_slots=CapabilityQuestionSlots(
                manual_question=parsed.manual_question,
            ),
        )
    if tool_name == "create_after_sales_draft":
        parsed = AfterSalesDraftArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="create_after_sales_ticket",
            arguments=parsed.model_dump(exclude_none=True),
            response_mode="after_sales_draft",
        )
    if tool_name == "request_human_handoff":
        parsed = HumanHandoffArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="create_human_handoff",
            arguments=parsed.model_dump(exclude_none=True),
            response_mode="human_handoff",
        )
    if tool_name == "clarify_request":
        EmptyArgs.model_validate(arguments)
        return CapabilitySuggestion(
            tool_name="none",
            response_mode="clarification",
        )
    raise ValueError(f"unsupported read-only capability tool: {tool_name}")


def _native_capability_tools() -> list[dict[str, Any]]:
    registry = get_tool_registry()
    definitions: list[tuple[str, str, type[BaseModel]]] = []
    if registry.get_tool("search_products", require_enabled=True) is not None:
        definitions.extend(
            [
                (
                    "search_products",
                    "查询商品列表。用户询问是否有某类商品或按明确条件找商品时使用。keyword必填。",
                    SearchProductListArgs,
                ),
                (
                    "get_product_catalog_detail",
                    "查询当前已确定商品的价格、库存或目录详情。product_code必填。",
                    ProductCodeArgs,
                ),
                (
                    "check_product_feature",
                    "核验当前已确定商品是否具有某项目录特征。product_code和feature必填。",
                    ProductFeatureArgs,
                ),
            ]
        )
    if registry.get_tool("recommend_products", require_enabled=True) is not None:
        definitions.append(
            (
                "recommend_products",
                "用户明确要求推荐、换一个或其他选择时使用。keyword必填。",
                RecommendProductArgs,
            )
        )
    if registry.get_tool("compare_products", require_enabled=True) is not None:
        definitions.append(
            (
                "compare_products",
                "比较用户明确指定的2至5个商品。product_codes必填。",
                CompareProductArgs,
            )
        )
    if registry.get_tool("query_order", require_enabled=True) is not None:
        definitions.extend(
            [
                ("list_orders", "查询当前用户订单列表。无参数。", EmptyArgs),
                (
                    "get_order_detail",
                    "查询指定订单详情。order_ref必填。",
                    OrderRefArgs,
                ),
            ]
        )
    if registry.get_tool("query_logistics", require_enabled=True) is not None:
        definitions.append(
            (
                "query_logistics",
                "查询指定可信订单的物流状态。order_ref必填。",
                OrderRefArgs,
            )
        )
    if registry.get_tool("knowledge_search", require_enabled=True) is not None:
        definitions.extend(
            [
                (
                    "search_product_manual",
                    "查询当前已确定商品说明书中的蓝牙、连接、充电、兼容、按键、使用方法、保修或用途能力。product_code和manual_question必填。",
                    ProductManualArgs,
                ),
                (
                    "select_product_for_manual",
                    "用户明确询问某类或某款商品的说明书事实，但当前没有可信商品编码时，先按商品条件验证一个商品，成功后再查其说明书。keyword和manual_question必填。",
                    ProductManualSelectionArgs,
                ),
            ]
        )
    if registry.get_tool("create_after_sales_ticket", require_enabled=True) is not None:
        definitions.append(
            (
                "create_after_sales_draft",
                "提出售后草稿候选。只能创建draft，绝不能确认提交。已知字段按Schema填写，缺失字段留空。",
                AfterSalesDraftArgs,
            )
        )
    if registry.get_tool("create_human_handoff", require_enabled=True) is not None:
        definitions.append(
            (
                "request_human_handoff",
                "用户明确要求人工客服时提出转人工候选。不得伪造订单号或手机号。message必填。",
                HumanHandoffArgs,
            )
        )
    definitions.append(
        (
            "clarify_request",
            "没有合适且安全的只读能力，或缺少可信实体时使用。无参数。",
            EmptyArgs,
        )
    )
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": args_model.model_json_schema(),
            },
        }
        for name, description, args_model in definitions
    ]


def _to_semantic_frame(
    suggestion: CapabilitySuggestion,
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
        product_code = arguments.get("product_code")
        if not isinstance(product_code, str):
            return None
        manual_question = suggestion.question_slots.manual_question or rewritten_query
        return SemanticFrame(
            intent="product_fact",
            slots={
                **slots,
                "question_predicate": _manual_question_predicate(manual_question),
            },
            references=[
                ReferenceExpression(text=product_code, explicit_code=product_code)
            ],
            question=manual_question,
            requires_manual_evidence=True,
        )
    if suggestion.tool_name in {"search_products", "recommend_products"}:
        if suggestion.response_mode == "manual_fact_with_selection":
            manual_question = suggestion.question_slots.manual_question or rewritten_query
            return SemanticFrame(
                intent="product_fact_with_selection",
                slots={
                    **slots,
                    "question_predicate": _manual_question_predicate(manual_question),
                },
                requested_count=1,
                question=manual_question,
                requires_manual_evidence=True,
            )
        if suggestion.response_mode in {
            "product_catalog_detail",
            "product_feature_match",
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
    if suggestion.tool_name == "create_after_sales_ticket":
        return SemanticFrame(intent="after_sales", slots=slots)
    if suggestion.tool_name == "create_human_handoff":
        return SemanticFrame(intent="handoff", slots=slots)
    return None


def _validate_suggested_arguments(
    suggestion: CapabilitySuggestion,
    state: CustomerServiceState,
) -> dict[str, Any]:
    if suggestion.tool_name == "none":
        _validate_response_contract(suggestion)
        if suggestion.arguments:
            raise ValueError("none tool must not contain arguments")
        return {}
    if suggestion.tool_name == "knowledge_search":
        _validate_response_contract(suggestion)
        if set(suggestion.arguments) != {"product_code"}:
            raise ValueError("knowledge_search requires only trusted product_code")
        product_code = suggestion.arguments.get("product_code")
        trusted = _trusted_product_codes(state)
        if not isinstance(product_code, str) or product_code not in trusted:
            raise ValueError("manual product_code is outside trusted visible batch")
        batch = state.product.active_batch
        item = (
            next(
                (
                    candidate
                    for candidate in batch.items
                    if candidate.product_code == product_code
                ),
                None,
            )
            if batch is not None
            else None
        )
        if item is None or item.primary_manual_document_id is None:
            raise ValueError("trusted active product manual is missing")
        return {"product_code": product_code}
    if suggestion.tool_name in {"create_after_sales_ticket", "create_human_handoff"}:
        _validate_response_contract(suggestion)
        return dict(suggestion.arguments)
    protected = _PROTECTED_ARGUMENTS & suggestion.arguments.keys()
    if protected:
        raise ValueError(f"protected tool arguments: {sorted(protected)}")
    tool = get_tool_registry().get_tool(suggestion.tool_name, require_enabled=True)
    if tool is None:
        raise ValueError(f"capability tool not found: {suggestion.tool_name}")
    validated = tool.args_schema.model_validate(suggestion.arguments).model_dump()
    _validate_response_contract(suggestion)
    product_code = suggestion.arguments.get("product_code")
    if isinstance(product_code, str) and suggestion.response_mode in {
        "product_catalog_detail",
        "product_feature_match",
    }:
        trusted_codes = _trusted_product_codes(state)
        if product_code not in trusted_codes:
            raise ValueError("suggested product_code is outside trusted visible batch")
    return {
        key: validated[key]
        for key in suggestion.arguments
        if key in validated
    }


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


def _trusted_product_codes(state: CustomerServiceState) -> set[str]:
    batch = state.product.active_batch
    return {item.product_code for item in batch.items} if batch is not None else set()


def _validate_response_contract(suggestion: CapabilitySuggestion) -> None:
    allowed = {
        "search_products": {
            "product_list",
            "product_catalog_detail",
            "product_feature_match",
        },
        "recommend_products": {
            "product_recommendation",
            "manual_fact_with_selection",
        },
        "compare_products": {"product_comparison"},
        "query_order": {"order_list", "order_detail"},
        "query_logistics": {"logistics_status"},
        "knowledge_search": {"manual_fact"},
        "create_after_sales_ticket": {"after_sales_draft"},
        "create_human_handoff": {"human_handoff"},
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
        } and not isinstance(suggestion.arguments.get("product_code"), str):
            raise ValueError(f"{suggestion.response_mode} requires product_code")
    if suggestion.tool_name == "recommend_products" and not _has_product_condition(
        suggestion.arguments
    ):
        raise ValueError("recommend_products requires at least one product query condition")
    if suggestion.response_mode == "order_list" and "order_ref" in suggestion.arguments:
        raise ValueError("order_list must not contain order_ref")
    if suggestion.response_mode == "order_detail" and not isinstance(
        suggestion.arguments.get("order_ref"), str
    ):
        raise ValueError("order_detail requires order_ref")
    if (
        suggestion.response_mode == "product_feature_match"
        and suggestion.question_slots.feature is None
    ):
        raise ValueError("product_feature_match requires feature")
    if (
        suggestion.response_mode in {"manual_fact", "manual_fact_with_selection"}
        and suggestion.question_slots.manual_question is None
    ):
        raise ValueError(f"{suggestion.response_mode} requires manual_question")


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


def _manual_question_predicate(question: str) -> str:
    if "蓝牙" in question or "连接" in question:
        return "bluetooth_connectivity"
    if "充电" in question:
        return "charging"
    if "兼容" in question:
        return "compatibility"
    if "按键" in question:
        return "buttons"
    if "保修" in question:
        return "warranty"
    if any(term in question for term in ("怎么用", "如何使用", "使用方法")):
        return "usage"
    if any(term in question for term in ("游戏", "办公", "适合")):
        return "use_case"
    return "features"
