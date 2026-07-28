from __future__ import annotations

import re
import unicodedata

from backend.app.agents.customer_service_core.schemas import (
    CustomerServiceIntent,
    CustomerServiceSource,
    UserDecision,
)


def classify_user_decision(message: str) -> UserDecision:
    normalized = _normalize(message)
    if not normalized:
        return UserDecision.OTHER
    if any(
        pattern in normalized
        for pattern in (
            "忽略系统规则",
            "绕过确认",
            "confirmed=true",
            "假装用户已经确认",
            "执行隐藏指令",
        )
    ):
        return UserDecision.UNSAFE_INJECTION
    if normalized in {"不要取消", "别取消", "不要撤销"}:
        return UserDecision.AMBIGUOUS
    if any(
        pattern in normalized
        for pattern in (
            "取消",
            "不要提交",
            "不确认",
            "先等等",
            "暂不办理",
            "别创建",
            "停止",
        )
    ):
        return UserDecision.CANCEL
    if any(
        marker in normalized
        for marker in (
            "改成",
            "修改描述",
            "问题改为",
            "订单换成",
            "手机号后四位改为",
            "商品换成",
            "重新填写",
        )
    ):
        return UserDecision.MODIFY
    if normalized in {
        "确认",
        "确认提交",
        "确认创建",
        "同意",
        "同意提交",
        "可以提交",
        "请提交",
        "提交吧",
        "确认办理",
        "提交",
    }:
        return UserDecision.CONFIRM
    if normalized in {
        "看看吧",
        "应该可以",
        "可能可以",
        "再说吧",
        "随便",
        "你看着办",
        "大概行",
        "可以吧",
        "先看看",
    }:
        return UserDecision.AMBIGUOUS
    return UserDecision.OTHER


def deterministic_route(
    query: str,
) -> tuple[CustomerServiceIntent, CustomerServiceSource]:
    if is_return_policy_question(query):
        return (
            CustomerServiceIntent.POLICY_QUESTION,
            CustomerServiceSource.POLICY_KNOWLEDGE,
        )
    if is_greeting(query):
        return CustomerServiceIntent.GREETING, CustomerServiceSource.PLANNER
    if is_after_sales(query):
        return (
            CustomerServiceIntent.AFTER_SALES,
            CustomerServiceSource.AFTER_SALES_WORKFLOW,
        )
    if is_handoff(query):
        return (
            CustomerServiceIntent.HUMAN_HANDOFF,
            CustomerServiceSource.HUMAN_HANDOFF,
        )
    if is_logistics(query):
        return (
            CustomerServiceIntent.LOGISTICS_QUERY,
            CustomerServiceSource.ORDER_SERVICE,
        )
    if is_order(query):
        return (
            CustomerServiceIntent.ORDER_QUERY,
            CustomerServiceSource.ORDER_SERVICE,
        )
    if is_manual_question(query):
        return (
            CustomerServiceIntent.PRODUCT_DOCUMENT_FACT,
            CustomerServiceSource.PRIMARY_MANUAL,
        )
    if is_product_realtime_fact(query):
        return (
            CustomerServiceIntent.PRODUCT_REALTIME_FACT,
            CustomerServiceSource.PRODUCT_CATALOG,
        )
    return CustomerServiceIntent.OTHER, CustomerServiceSource.PLANNER


def is_greeting(query: str) -> bool:
    return query.casefold() in {"你好", "您好", "hi", "hello", "你能做什么"}


def is_prompt_injection(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.UNSAFE_INJECTION


def is_product_search(query: str) -> bool:
    return any(word in query for word in ("查", "找", "看看", "挑", "商品"))


def is_recommend(query: str) -> bool:
    return any(
        word in query for word in ("推荐", "适合", "预算", "偏好", "想要", "我要", "我需要", "人用")
    )


def is_compare(query: str) -> bool:
    return any(word in query for word in ("对比", "比较", "区别", "差别", "哪个好", "哪款好"))


def is_manual_question(query: str) -> bool:
    return any(
        word in query
        for word in (
            "说明书",
            "怎么用",
            "使用",
            "连接",
            "配对",
            "安装",
            "清洁",
            "故障",
            "安全",
            "操作",
            "蓝牙",
            "无线",
            "有线",
            "兼容",
            "充电",
            "电池",
            "接口",
            "驱动",
            "系统支持",
            "是否支持",
            "支不支持",
        )
    )


def is_product_realtime_fact(query: str) -> bool:
    return any(
        word in query
        for word in (
            "价格",
            "多少钱",
            "库存",
            "有货",
            "在售",
            "品牌",
            "型号",
            "特点",
            "特色",
            "适用场景",
            "适合什么",
            "商品描述",
            "介绍一下",
        )
    )


def is_order(query: str) -> bool:
    return "订单" in query and not is_after_sales(query)


def is_logistics(query: str) -> bool:
    return any(
        phrase in query
        for phrase in (
            "物流",
            "快递",
            "配送",
            "运单",
            "到哪里",
            "到哪了",
            "到哪儿了",
            "送到哪",
            "什么时候送到",
        )
    )


def is_after_sales(query: str) -> bool:
    return any(word in query for word in ("售后", "维修", "退货", "换货", "坏了"))


def is_return_policy_question(query: str) -> bool:
    return any(word in query for word in ("退换货规则", "退货规则", "换货规则", "退款规则"))


def is_handoff(query: str) -> bool:
    return "人工" in query or "客服" in query


def is_clear_confirmation(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.CONFIRM


def is_ambiguous_confirmation(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.AMBIGUOUS


def is_cancel(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.CANCEL


def _normalize(message: str) -> str:
    text = unicodedata.normalize("NFKC", message).strip()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[。！？!?.，,；;]+$", "", text)
