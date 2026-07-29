"""Layer 2: 确定性控制器。

映射表 + 参数校验 + 指代消解 + 追问判定 + confirm 流程。
纯代码，不调 LLM。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.agents.customer_service_v2.session import ProductRef, SessionState
from backend.app.agents.customer_service_v2.understanding import (
    IntentEnum,
    UnderstandingOutput,
)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ControllerDecision:
    """控制器输出。"""

    tool: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    ask: str | None = None  # 追问/确认文本
    direct_answer: str | None = None  # 直接回答（greeting 等）


@dataclass
class ActionSpec:
    """意图 → 动作映射规格。"""

    tool: str
    required: list[str] = field(default_factory=list)
    confirm: bool = False
    pre_hook: Callable[..., ControllerDecision | None] | None = None


# ---------------------------------------------------------------------------
# 映射表
# ---------------------------------------------------------------------------

ACTION_TABLE: dict[IntentEnum, ActionSpec] = {
    IntentEnum.RECOMMEND: ActionSpec(tool="recommend_products"),
    IntentEnum.SEARCH: ActionSpec(tool="search_products"),
    IntentEnum.COMPARE: ActionSpec(tool="compare_products", required=["targets>=2"]),
    IntentEnum.FACT: ActionSpec(tool="knowledge_search", required=["target"]),
    IntentEnum.ORDER: ActionSpec(tool="query_order", required=["order_ref"]),
    IntentEnum.LOGISTICS: ActionSpec(tool="query_logistics", required=["order_ref"]),
    IntentEnum.AFTER_SALES: ActionSpec(
        tool="create_after_sales_ticket",
        required=["order_ref", "phone_last4", "reason"],
        confirm=True,
    ),
    IntentEnum.HANDOFF: ActionSpec(tool="create_human_handoff"),
    IntentEnum.GREETING: ActionSpec(tool="__direct_answer__"),
    IntentEnum.OTHER: ActionSpec(tool="__clarify__"),
}

_GREETING_ANSWER = "你好，我可以协助查询商品、说明书、订单物流、售后和转人工。"
_CLARIFY_ANSWER = "我暂时无法准确理解您的需求，请补充要查询的商品、订单或具体问题。"


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def decide(understanding: UnderstandingOutput, session: SessionState) -> ControllerDecision:
    """控制器主入口：理解结果 → 执行决策。"""
    # 0. 安全拦截
    if understanding.intent == IntentEnum.BLOCKED:
        return ControllerDecision(direct_answer="我不能忽略系统规则或绕过工具确认流程。")

    # 1. 确认流程拦截
    if understanding.intent == IntentEnum.CONFIRM and session.pending_action:
        return _execute_pending(session)
    if understanding.intent == IntentEnum.CANCEL and session.pending_action:
        session.pending_action = None
        return ControllerDecision(direct_answer="已取消，未创建工单。")

    # 2. 查表
    spec = ACTION_TABLE.get(understanding.intent)
    if spec is None:
        return ControllerDecision(ask=_CLARIFY_ANSWER)

    # 3. 直接回答类
    if spec.tool == "__direct_answer__":
        return ControllerDecision(direct_answer=_GREETING_ANSWER)
    if spec.tool == "__clarify__":
        return ControllerDecision(ask=_CLARIFY_ANSWER)

    # 4. 指代消解
    resolved_targets = resolve_refs(understanding.target_refs, session.active_products)

    # 5. 参数组装
    params = _build_tool_params(understanding, session, resolved_targets)

    # 6. required 校验
    missing = _check_required(spec.required, params, resolved_targets)
    if missing:
        return ControllerDecision(ask=_generate_clarification(missing))

    # 7. confirm 拦截
    if spec.confirm:
        session.pending_action = {"tool": spec.tool, "params": params}
        return ControllerDecision(ask=_generate_confirmation_summary(params))

    # 8. pre_hook
    if spec.pre_hook is not None:
        hook_result = spec.pre_hook(params, session)
        if hook_result is not None:
            return hook_result

    # 9. 执行
    return ControllerDecision(tool=spec.tool, params=params)


# ---------------------------------------------------------------------------
# 指代消解
# ---------------------------------------------------------------------------

_ORDINAL_RE = re.compile(r"第\s*([一二三四五1-5])\s*(?:个|款|件|只|笔)")
_ORDINAL_MAP = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4}


def resolve_refs(refs: list[str], active: list[ProductRef]) -> list[ProductRef]:
    """将用户指代文本解析为具体商品引用。"""
    if not refs or not active:
        return []
    results: list[ProductRef] = []
    for ref_text in refs:
        matched = _match_single_ref(ref_text, active)
        if matched is not None:
            results.append(matched)
    return results


def _match_single_ref(ref_text: str, active: list[ProductRef]) -> ProductRef | None:
    """匹配单个指代。"""
    # "都"/"所有"/"全部" → 不做单个匹配，由调用方处理
    if ref_text in ("都", "所有", "全部"):
        return active[0] if active else None

    # 序号: "第一个"/"第2款"
    ordinal_match = _ORDINAL_RE.search(ref_text)
    if ordinal_match:
        raw = ordinal_match.group(1)
        index = _ORDINAL_MAP.get(raw, int(raw) - 1 if raw.isdigit() else -1)
        if 0 <= index < len(active):
            return active[index]
        return None

    # 纯数字序号: "1"/"2"
    if ref_text.strip().isdigit():
        index = int(ref_text.strip()) - 1
        if 0 <= index < len(active):
            return active[index]
        return None

    # 名称/型号模糊匹配
    ref_lower = ref_text.lower()
    for product in active:
        if (
            ref_lower in product.name.lower()
            or ref_lower in product.product_code.lower()
        ):
            return product

    return None


# ---------------------------------------------------------------------------
# 参数组装
# ---------------------------------------------------------------------------


def _build_tool_params(
    understanding: UnderstandingOutput,
    session: SessionState,
    resolved_targets: list[ProductRef],
) -> dict[str, Any]:
    """从理解结果 + 会话状态组装 tool 参数。"""
    params: dict[str, Any] = {}

    # 合并 filters（增量）
    filters = dict(session.last_filters)
    _apply_filter_operations(filters, understanding)

    # 商品搜索/推荐参数
    if understanding.keyword:
        filters["keyword"] = understanding.keyword
    if understanding.category:
        filters["category"] = understanding.category
    if understanding.brand:
        filters["brand"] = understanding.brand
    if understanding.model:
        filters["model"] = understanding.model
    if understanding.price_min is not None:
        filters["price_min"] = understanding.price_min
    if understanding.price_max is not None:
        filters["price_max"] = understanding.price_max
    if understanding.use_cases:
        filters["required_use_cases"] = understanding.use_cases
    if understanding.features:
        filters["required_features"] = understanding.features

    # 清理 None 值
    params.update({k: v for k, v in filters.items() if v is not None})

    # 目标商品
    if resolved_targets:
        params["target_product_codes"] = [p.product_code for p in resolved_targets]
        if len(resolved_targets) == 1:
            params["target"] = resolved_targets[0].product_code

    # 订单
    order_ref = understanding.order_ref or session.active_order_ref
    if order_ref:
        params["order_ref"] = order_ref
        params["order_no"] = order_ref
    if understanding.phone_last4:
        params["customer_phone_last4"] = understanding.phone_last4
    if understanding.after_sales_reason:
        params["reason"] = understanding.after_sales_reason

    # 推荐数量
    if understanding.recommendation_count:
        params["page_size"] = understanding.recommendation_count

    return params


def _apply_filter_operations(filters: dict[str, Any], understanding: UnderstandingOutput) -> None:
    """应用 filter_operations（SET/REMOVE）到 filters。"""
    for key, op in understanding.filter_operations.items():
        if op == "REMOVE":
            filters.pop(key, None)
        elif op == "SET":
            # SET 时新值从 understanding 对应字段取
            new_value = getattr(understanding, key, None)
            if new_value is not None:
                filters[key] = new_value


# ---------------------------------------------------------------------------
# required 校验
# ---------------------------------------------------------------------------


def _check_required(
    required: list[str],
    params: dict[str, Any],
    resolved_targets: list[ProductRef],
) -> list[str]:
    """检查必填参数，返回缺失列表。"""
    missing: list[str] = []
    for req in required:
        if req == "targets>=2":
            if len(resolved_targets) < 2:
                missing.append("至少两个对比商品")
        elif req == "target":
            if not params.get("target") and not params.get("target_product_codes"):
                missing.append("具体商品")
        elif req == "order_ref":
            if not params.get("order_ref"):
                missing.append("订单号")
        elif req == "phone_last4":
            if not params.get("customer_phone_last4"):
                missing.append("手机尾号后四位")
        elif req == "reason":
            if not params.get("reason"):
                missing.append("售后原因")
        else:
            if not params.get(req):
                missing.append(req)
    return missing


def _generate_clarification(missing: list[str]) -> str:
    """生成追问文本。"""
    items = "、".join(missing)
    return f"请提供{items}，我才能为您处理。"


def _generate_confirmation_summary(params: dict[str, Any]) -> str:
    """生成确认摘要。"""
    parts = ["请确认以下售后工单信息："]
    if params.get("order_no"):
        parts.append(f"- 订单号：{params['order_no']}")
    if params.get("customer_phone_last4"):
        parts.append(f"- 手机尾号：{params['customer_phone_last4']}")
    if params.get("reason"):
        parts.append(f"- 问题描述：{params['reason']}")
    parts.append('确认无误请回复"确认提交"，如需修改请说明。')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 确认执行
# ---------------------------------------------------------------------------


def _execute_pending(session: SessionState) -> ControllerDecision:
    """执行待确认的操作。"""
    pending = session.pending_action
    session.pending_action = None
    if not pending:
        return ControllerDecision(ask="当前没有待确认的操作。")
    tool = pending.get("tool", "")
    params = pending.get("params", {})
    params["confirmed"] = True
    params["action"] = "confirm"
    return ControllerDecision(tool=tool, params=params)
