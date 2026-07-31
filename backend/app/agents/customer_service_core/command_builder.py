from __future__ import annotations

from typing import Any

from backend.app.agents.customer_service_core.contracts import (
    CompareProductsCommand,
    ConfirmAfterSalesCommand,
    CreateAfterSalesDraftCommand,
    CreateHumanHandoffCommand,
    CustomerServiceCommand,
    CustomerServiceState,
    QueryLogisticsCommand,
    QueryOrderCommand,
    RecommendProductsCommand,
    SearchProductsCommand,
    SemanticFrame,
    StatePreview,
)
from backend.app.agents.customer_service_core.entity_resolver import (
    EntityResolution,
    ResolutionStatus,
    resolve_product_reference,
)
from pydantic import BaseModel, ConfigDict


class CommandBuildResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: CustomerServiceCommand | None = None
    direct_answer: str | None = None
    clarification: str | None = None
    resolution: EntityResolution | None = None
    expected_result_type: str = "unknown"
    state_action: str | None = None


def build_command(
    *,
    frame: SemanticFrame,
    preview: StatePreview,
    runtime: dict[str, Any],
) -> CommandBuildResult:
    state = preview.proposed_state
    knowledge_base_id = _trusted_knowledge_base_id(runtime)
    if frame.intent == "blocked":
        return CommandBuildResult(direct_answer="我不能忽略系统规则或绕过工具确认流程。")
    if frame.intent == "greeting":
        return CommandBuildResult(
            direct_answer="你好，我可以协助查询商品、说明书、订单物流、售后和转人工。"
        )
    if frame.intent == "other":
        return CommandBuildResult(clarification="请说明您要查询的商品、订单或具体问题。")
    if frame.intent == "product_query_confirmation":
        category = frame.slots.get("category")
        keyword = frame.slots.get("keyword")
        if not isinstance(category, str) or not isinstance(keyword, str):
            return CommandBuildResult(clarification="请说明需要重新查询的商品品类。")
        return CommandBuildResult(
            direct_answer=(
                f"抱歉，我目前不清楚您指的是哪款{keyword}。需要我重新为您查询{keyword}吗？"
            ),
            state_action="set_pending_product_query",
        )
    if frame.intent == "confirm_pending_product_query":
        pending = state.product.pending_query
        if pending is None or frame.slots.get("confirmation") is not True:
            return CommandBuildResult(clarification="当前没有等待确认的商品查询。")
        filters = dict(state.product.filters)
        filters.update({"category": pending.category, "keyword": pending.keyword})
        return CommandBuildResult(
            command=RecommendProductsCommand(
                filters=filters,
                page_size=pending.requested_count,
                knowledge_base_id=knowledge_base_id,
            ),
            expected_result_type="products",
        )
    if frame.intent == "cancel":
        if state.pending_after_sales is None:
            return CommandBuildResult(direct_answer="当前没有待确认的售后申请。")
        return CommandBuildResult(direct_answer="已取消，未提交售后工单。")
    if frame.intent == "confirm":
        pending = state.pending_after_sales
        if pending is None:
            return CommandBuildResult(clarification="当前没有待确认的售后申请。")
        return CommandBuildResult(
            command=ConfirmAfterSalesCommand(
                order_no=pending.order_no,
                customer_phone_last4=pending.customer_phone_last4,
                draft_id=pending.draft_id,
                operation_id=pending.operation_id,
            ),
            expected_result_type="after_sales_confirm",
        )
    if frame.intent in {"recommend_products", "search_products"}:
        filters = dict(state.product.filters)
        filters.pop("confirmed_pending_product_query", None)
        if not _has_product_query_condition(filters):
            return CommandBuildResult(clarification="请说明您需要推荐的商品名称、类型或具体需求。")
        page_size = frame.requested_count or 1
        command_type = (
            RecommendProductsCommand
            if frame.intent == "recommend_products"
            else SearchProductsCommand
        )
        if command_type is RecommendProductsCommand:
            return CommandBuildResult(
                command=RecommendProductsCommand(
                    filters=filters,
                    page_size=page_size,
                    knowledge_base_id=knowledge_base_id,
                ),
                expected_result_type="products",
            )
        return CommandBuildResult(
            command=SearchProductsCommand(
                keyword=filters.pop("keyword", None),
                category=filters.pop("category", None),
                filters=filters,
                page_size=20,
                knowledge_base_id=knowledge_base_id,
            ),
            expected_result_type="products",
        )
    if frame.intent == "compare_products":
        product_codes: list[str] = []
        for reference in frame.references:
            resolution = resolve_product_reference(reference, state)
            if resolution.status == ResolutionStatus.RESOLVED:
                product_codes.extend(resolution.product_codes)
            elif (
                resolution.status == ResolutionStatus.VERIFICATION_REQUIRED
                and reference.explicit_code
            ):
                product_codes.append(reference.explicit_code)
            else:
                return CommandBuildResult(
                    clarification="请明确选择两个要对比的商品。",
                    resolution=resolution,
                )
        product_codes = list(dict.fromkeys(product_codes))
        if len(product_codes) < 2:
            return CommandBuildResult(clarification="请明确选择两个要对比的商品。")
        return CommandBuildResult(
            command=CompareProductsCommand(
                product_codes=product_codes[:5],
                knowledge_base_id=knowledge_base_id,
            ),
            expected_result_type="product_comparison",
        )
    if frame.intent == "product_fact":
        resolution = _resolve_product(frame, state)
        if resolution.status == ResolutionStatus.AMBIGUOUS:
            return CommandBuildResult(
                clarification="请明确选择要查询的商品。",
                resolution=resolution,
            )
        if resolution.status == ResolutionStatus.NOT_FOUND:
            return CommandBuildResult(
                clarification="当前候选中没有找到您指的商品。",
                resolution=resolution,
            )
        if resolution.status == ResolutionStatus.VERIFICATION_REQUIRED:
            query = resolution.verification_query
            return CommandBuildResult(
                command=SearchProductsCommand(
                    product_code=query.get("product_code"),
                    keyword=query.get("keyword"),
                    page_size=5,
                    knowledge_base_id=knowledge_base_id,
                ),
                resolution=resolution,
                expected_result_type=(
                    "product_verification_for_manual"
                    if frame.requires_manual_evidence
                    else "product_verification_for_catalog"
                ),
            )
        product_code = resolution.product_codes[0]
        return CommandBuildResult(
            command=SearchProductsCommand(
                product_code=product_code,
                page_size=1,
                knowledge_base_id=knowledge_base_id,
            ),
            resolution=resolution,
            expected_result_type=(
                "product_verification_for_manual"
                if frame.requires_manual_evidence
                else "product_catalog_fact"
            ),
        )
    if frame.intent == "product_fact_with_selection":
        filters = dict(state.product.filters)
        return CommandBuildResult(
            command=RecommendProductsCommand(
                filters=filters,
                page_size=1,
                knowledge_base_id=knowledge_base_id,
            ),
            expected_result_type=(
                "product_selection_for_manual_fact"
                if frame.requires_manual_evidence
                else "product_selection_for_catalog_fact"
            ),
        )
    if frame.intent == "order":
        order_ref = _resolved_order_ref(frame, state)
        return CommandBuildResult(
            command=QueryOrderCommand(order_ref=order_ref),
            expected_result_type="order",
        )
    if frame.intent == "logistics":
        order_ref = _resolved_order_ref(frame, state)
        if order_ref is None:
            if not state.order_candidates:
                return CommandBuildResult(
                    command=QueryOrderCommand(),
                    expected_result_type="order_list_for_logistics",
                )
            return CommandBuildResult(clarification="请从当前订单列表中选择要查询物流的订单。")
        if _requires_order_verification(frame, runtime, order_ref):
            return CommandBuildResult(
                command=QueryOrderCommand(order_ref=order_ref),
                expected_result_type="order_verification_for_logistics",
            )
        return CommandBuildResult(
            command=QueryLogisticsCommand(order_ref=order_ref),
            expected_result_type="logistics",
        )
    if frame.intent == "after_sales":
        order_ref = _resolved_order_ref(frame, state)
        if order_ref is None:
            if not state.order_candidates:
                return CommandBuildResult(
                    command=QueryOrderCommand(),
                    expected_result_type="order_list_for_after_sales",
                )
            return CommandBuildResult(clarification="请先选择需要售后的订单。")
        phone_last4 = frame.slots.get("phone_last4")
        if not isinstance(phone_last4, str):
            return CommandBuildResult(clarification="请提供订单联系人手机号后四位。")
        description = str(frame.slots.get("issue_description") or "")
        if len(description.strip()) < 5:
            return CommandBuildResult(clarification="请说明具体售后原因。")
        if _requires_order_verification(frame, runtime, order_ref):
            return CommandBuildResult(
                command=QueryOrderCommand(order_ref=order_ref),
                expected_result_type="order_verification_for_after_sales",
            )
        return CommandBuildResult(
            command=CreateAfterSalesDraftCommand(
                order_no=order_ref,
                customer_phone_last4=phone_last4,
                issue_type=frame.slots.get("issue_type", "other"),
                issue_description=description,
            ),
            expected_result_type="after_sales_draft",
        )
    if frame.intent == "handoff":
        order_ref = _resolved_order_ref(frame, state)
        phone_last4 = frame.slots.get("phone_last4")
        if order_ref is None or not isinstance(phone_last4, str):
            return CommandBuildResult(clarification="请提供需要转人工处理的订单和手机号后四位。")
        if _requires_order_verification(frame, runtime, order_ref):
            return CommandBuildResult(
                command=QueryOrderCommand(order_ref=order_ref),
                expected_result_type="order_verification_for_handoff",
            )
        return CommandBuildResult(
            command=CreateHumanHandoffCommand(
                order_no=order_ref,
                customer_phone_last4=phone_last4,
                message=str(frame.slots.get("message") or "用户请求人工协助"),
            ),
            expected_result_type="handoff",
        )
    return CommandBuildResult(clarification="当前请求还需要更多信息。")


def _trusted_knowledge_base_id(runtime: dict[str, Any]) -> int | None:
    knowledge_base_id = runtime.get("knowledge_base_id")
    allowed = {
        value for value in runtime.get("allowed_knowledge_base_ids", []) if isinstance(value, int)
    }
    if isinstance(knowledge_base_id, int) and knowledge_base_id in allowed:
        return knowledge_base_id
    return None


def _has_product_query_condition(filters: dict[str, Any]) -> bool:
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
    return any(filters.get(field) not in {None, ""} for field in scalar_fields) or any(
        isinstance(filters.get(field), list) and bool(filters[field]) for field in list_fields
    )


def _resolve_product(
    frame: SemanticFrame,
    state: CustomerServiceState,
) -> EntityResolution:
    if frame.references:
        return resolve_product_reference(frame.references[0], state)
    latest = state.product.active_batch
    if latest and state.product.active_product_code:
        item = next(
            (
                candidate
                for candidate in latest.items
                if candidate.product_code == state.product.active_product_code
            ),
            None,
        )
        if item is not None:
            return EntityResolution(
                status=ResolutionStatus.RESOLVED,
                product_codes=[item.product_code],
                candidates=[item],
            )
    if latest and len(latest.items) == 1:
        item = latest.items[0]
        return EntityResolution(
            status=ResolutionStatus.RESOLVED,
            product_codes=[item.product_code],
            candidates=[item],
        )
    return EntityResolution(
        status=ResolutionStatus.AMBIGUOUS,
        candidates=latest.items if latest else [],
    )


def _resolved_order_ref(
    frame: SemanticFrame,
    state: CustomerServiceState,
) -> str | None:
    explicit = frame.slots.get("order_ref")
    if isinstance(explicit, str):
        return explicit
    if frame.references and frame.references[0].ordinal is not None:
        index = frame.references[0].ordinal
        if index < len(state.order_candidates):
            item = state.order_candidates[index]
            value = item.get("order_ref") or item.get("order_no")
            return str(value) if value else None
    return state.active_order_ref


def _requires_order_verification(
    frame: SemanticFrame,
    runtime: dict[str, Any],
    order_ref: str,
) -> bool:
    explicit = frame.slots.get("order_ref")
    return (
        isinstance(explicit, str)
        and explicit == order_ref
        and runtime.get("verified_order_ref") != order_ref
    )
