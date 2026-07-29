"""V2 PlannerStrategy 适配器。

对接 LangGraph 运行时，将 V2 三层架构包装为 BaseAgentPlannerStrategy。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from backend.app.agents.customer_service_v2.controller import (
    ControllerDecision,
    decide,
)
from backend.app.agents.customer_service_v2.rules import quick_understand
from backend.app.agents.customer_service_v2.session import (
    ProductRef,
    load_session,
    save_session,
)
from backend.app.agents.customer_service_v2.understanding import (
    IntentEnum,
    UnderstandingOutput,
    llm_understand,
)
from backend.app.agents.langgraph.tool_calling import (
    AgentDecision,
    AgentToolCall,
    BaseAgentPlannerStrategy,
)

logger = logging.getLogger(__name__)


class CustomerServiceV2Strategy(BaseAgentPlannerStrategy):
    """V2 客服 Planner：分层理解 + 确定性控制器。"""

    name = "customer_service_v2"

    async def adecide(self, state: Any) -> AgentDecision:
        # Tool 已执行过（observations 非空）→ 交给 FinalNode 生成回答
        if state.get("observations"):
            return AgentDecision(action="final", content=None)

        query = str(state.get("query") or "").strip()
        metadata = state.setdefault("metadata", {})
        session = load_session(metadata)

        # Layer 1a: 规则快速路径
        understanding = quick_understand(query, session)

        # Layer 1b: LLM（仅规则未命中时）
        if understanding is None:
            understanding = await llm_understand(
                query, session, state.get("messages", [])
            )

        # Layer 2: 确定性控制器
        decision = decide(understanding, session)

        # 更新会话状态
        _update_session_after_decision(session, decision, understanding)
        save_session(metadata, session)

        # 记录 debug 信息
        metadata.setdefault("customer_service", {})["v2_understanding"] = (
            understanding.model_dump(mode="json")
        )

        # 转换为 AgentDecision
        return _to_agent_decision(decision)


def _to_agent_decision(decision: ControllerDecision) -> AgentDecision:
    """将控制器决策转为 LangGraph AgentDecision。"""
    # 追问 / 确认
    if decision.ask:
        return AgentDecision(action="final", content=decision.ask)

    # 直接回答（greeting / blocked / cancel）
    if decision.direct_answer:
        return AgentDecision(action="final", content=decision.direct_answer)

    # Tool 执行
    if decision.tool:
        tool_call_id = f"cs_v2_{uuid4().hex}"
        return AgentDecision(
            action="tool_calls",
            tool_calls=[
                AgentToolCall(
                    id=tool_call_id,
                    tool_name=decision.tool,
                    arguments=decision.params,
                )
            ],
        )

    # 兜底
    return AgentDecision(
        action="final",
        content="我暂时无法准确理解您的需求，请补充要查询的商品、订单或具体问题。",
    )


def _update_session_after_decision(
    session: Any,
    decision: ControllerDecision,
    understanding: UnderstandingOutput,
) -> None:
    """执行后更新会话状态（filters 继承等）。"""
    # 推荐/搜索执行后，保存 filters 供下轮继承
    if understanding.intent in (IntentEnum.RECOMMEND, IntentEnum.SEARCH):
        if decision.tool and decision.params:
            # 只保留筛选相关的 key
            filter_keys = {
                "keyword",
                "brand",
                "category",
                "model",
                "price_min",
                "price_max",
                "required_features",
                "preferred_features",
                "required_use_cases",
                "preferred_use_cases",
            }
            session.last_filters = {
                k: v for k, v in decision.params.items() if k in filter_keys
            }

    # 订单引用持久化
    if understanding.order_ref:
        session.active_order_ref = understanding.order_ref


def update_session_products(
    metadata: dict[str, Any],
    products: list[dict[str, Any]],
) -> None:
    """Tool 执行后更新候选商品列表（由外部调用）。"""
    session = load_session(metadata)
    session.active_products = [
        ProductRef(
            ref=str(item.get("ref", str(i))),
            name=str(item.get("name", "")),
            product_code=str(item.get("product_code", "")),
            category=item.get("category"),
        )
        for i, item in enumerate(products[:5], start=1)
    ]
    save_session(metadata, session)
