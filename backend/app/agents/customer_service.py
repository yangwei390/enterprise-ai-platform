from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from enum import StrEnum
from time import monotonic
from typing import Any
from uuid import uuid4

from backend.app.agents.customer_service_contract import (
    CUSTOMER_SERVICE_AGENT_ID,
    CUSTOMER_SERVICE_CONFIRMED_STATUS,
    CUSTOMER_SERVICE_CONFIRMING_STATUS,
    CUSTOMER_SERVICE_PENDING_KEY,
    CUSTOMER_SERVICE_PENDING_STATUS,
    CUSTOMER_SERVICE_TOOL_ALLOWLIST,
)
from backend.app.agents.langgraph.tool_calling import (
    AgentDecision,
    AgentToolCall,
    BaseAgentPlannerStrategy,
)
from backend.app.tools.base import ToolResult


class UserDecision(StrEnum):
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    AMBIGUOUS = "AMBIGUOUS"
    OTHER = "OTHER"
    UNSAFE_INJECTION = "UNSAFE_INJECTION"


class _PendingCoordinator:
    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._locks: OrderedDict[str, threading.RLock] = OrderedDict()
        self._active: dict[str, tuple[str, int, float]] = {}
        self._completed: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._max_locks = 1024
        self._max_completed = 2048
        self._reservation_ttl_seconds = 60.0

    def lock_for(self, session_id: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[session_id] = lock
            else:
                self._locks.move_to_end(session_id)
            self._cleanup_expired_locked()
            if len(self._locks) > self._max_locks:
                for old_session_id in list(self._locks):
                    if old_session_id in self._active:
                        continue
                    self._locks.pop(old_session_id, None)
                    if len(self._locks) <= self._max_locks:
                        break
            return lock

    def reserve(self, session_id: str, operation_id: str, version: int) -> bool:
        with self.lock_for(session_id):
            with self._guard:
                self._cleanup_expired_locked()
                if (session_id, operation_id) in self._completed:
                    self._completed.move_to_end((session_id, operation_id))
                    return False
                if session_id in self._active:
                    return False
                self._active[session_id] = (
                    operation_id,
                    version,
                    monotonic() + self._reservation_ttl_seconds,
                )
                return True

    def finalize(self, session_id: str, operation_id: str, *, success: bool) -> None:
        with self.lock_for(session_id):
            with self._guard:
                active = self._active.get(session_id)
                if active and active[0] == operation_id:
                    self._active.pop(session_id, None)
                if success:
                    self._completed[(session_id, operation_id)] = None
                    self._completed.move_to_end((session_id, operation_id))
                    while len(self._completed) > self._max_completed:
                        self._completed.popitem(last=False)

    def active_count(self) -> int:
        with self._guard:
            self._cleanup_expired_locked()
            return len(self._active)

    def _cleanup_expired_locked(self) -> None:
        now = monotonic()
        expired = [
            session_id
            for session_id, (_, _, expires_at) in self._active.items()
            if expires_at <= now
        ]
        for session_id in expired:
            self._active.pop(session_id, None)


_PENDING_COORDINATOR = _PendingCoordinator()


class CustomerServicePlannerStrategy(BaseAgentPlannerStrategy):
    name = "customer_service_rules"

    async def adecide(self, state: Any) -> AgentDecision:
        query = str(state.get("query") or "").strip()
        metadata = state.setdefault("metadata", {})
        metadata["runtime_turn_id"] = current_runtime_turn_id(state)
        observations = state.get("observations", [])

        if _is_greeting(query):
            return _final("你好，我可以协助查询模拟商品、说明书、订单物流、模拟售后和模拟转人工。")

        pending = _pending_after_sales(metadata)
        if pending is not None:
            decision = classify_user_decision(latest_user_message(state) or query)
            if decision == UserDecision.CANCEL:
                _set_pending_after_sales(metadata, None)
                return _final("已取消本次模拟售后确认，未创建工单。")
            if decision == UserDecision.UNSAFE_INJECTION:
                return _final("我不能忽略系统规则或绕过工具确认流程。")
            if decision == UserDecision.MODIFY or _context_changed(pending, query):
                _set_pending_after_sales(metadata, None)
                return _after_sales_draft_or_clarify(query)
            if decision == UserDecision.CONFIRM:
                return _tool_decision(
                    "create_after_sales_ticket",
                    {
                        "action": "confirm",
                        "order_no": pending["order_no"],
                        "customer_phone_last4": pending["customer_phone_last4"],
                        "draft_id": pending["draft_id"],
                        "operation_id": pending["operation_id"],
                        "confirmed": True,
                    },
                )
            if _is_after_sales(query):
                _set_pending_after_sales(metadata, None)
                return _after_sales_draft_or_clarify(query)
            if decision == UserDecision.AMBIGUOUS:
                return _final("请明确回复“确认提交”后，我才能创建模拟售后工单。")

        if _last_tool_name(observations) == "search_products" and _is_manual_question(query):
            return _manual_followup_decision(state, observations[-1])
        if _last_tool_name(observations) == "knowledge_search":
            return _knowledge_final(state, observations[-1])
        if _last_tool_name(observations) == "create_after_sales_ticket":
            return _after_sales_final(metadata, observations[-1])
        if _last_tool_name(observations) == "query_order" and _is_logistics(query):
            order = _extract_order_fields(query)
            if order is None:
                return _final("请提供订单号和手机号后四位后再查询物流。")
            return _tool_decision("query_logistics", order)
        if _last_tool_name(observations) in {
            "search_products",
            "recommend_products",
            "compare_products",
            "query_order",
            "query_logistics",
            "create_human_handoff",
        }:
            return _observation_final(observations[-1])

        if _is_prompt_injection(query):
            return _final("我不能忽略系统规则或绕过工具确认流程。")
        if _is_after_sales(query):
            return _after_sales_draft_or_clarify(query)
        if _is_handoff(query):
            order = _extract_order_fields(query)
            if order is None:
                return _final("请提供订单号和手机号后四位后再创建模拟转人工记录。")
            return _tool_decision(
                "create_human_handoff",
                {
                    **order,
                    "reason": "customer_request",
                    "message": query[:500],
                },
            )
        if _is_logistics(query):
            order = _extract_order_fields(query)
            if order is None:
                return _final("请提供订单号和手机号后四位后再查询物流。")
            return _tool_decision("query_order", order)
        if _is_order(query):
            order = _extract_order_fields(query)
            if order is None:
                return _final("请提供订单号和手机号后四位后再查询订单。")
            return _tool_decision("query_order", order)
        if _is_manual_question(query):
            return _tool_decision(
                "search_products",
                _product_query_args(state, query, page_size=5),
            )
        if _is_compare(query):
            codes = _extract_product_codes(query)
            if len(codes) < 2:
                return _final("请提供至少两个明确的商品编码后再对比。")
            args: dict[str, Any] = {"product_codes": codes}
            if isinstance(state.get("knowledge_base_id"), int):
                args["knowledge_base_id"] = state["knowledge_base_id"]
            return _tool_decision("compare_products", args)
        if _is_recommend(query):
            return _tool_decision(
                "recommend_products",
                _product_query_args(state, query, page_size=3),
            )
        if _is_product_search(query):
            return _tool_decision(
                "search_products",
                _product_query_args(state, query, page_size=3),
            )

        return _final("我可以处理模拟商品、说明书、订单物流、售后和转人工相关问题。")


def evaluate_customer_service_tool_policy(
    *,
    state: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolResult | None:
    if state.get("metadata", {}).get("agent_id") != CUSTOMER_SERVICE_AGENT_ID:
        return None
    if tool_name != "create_after_sales_ticket":
        return None
    action = arguments.get("action", "draft")
    metadata = state.setdefault("metadata", {})
    if action == "draft":
        if _conversation_session_id(state) is None:
            return _blocked_tool_result(
                tool_name,
                "after_sales_conversation_required",
            )
        return None
    if action != "confirm":
        return _blocked_tool_result(tool_name, "unsupported_after_sales_action")
    session_id = _conversation_session_id(state)
    if session_id is None:
        return _blocked_tool_result(tool_name, "after_sales_conversation_required")
    latest_message = latest_user_message(state)
    if latest_message is None:
        return _blocked_tool_result(tool_name, "after_sales_user_confirmation_missing")
    decision = classify_user_decision(latest_message)
    if decision == UserDecision.CANCEL:
        _set_pending_after_sales(metadata, None)
        return _blocked_tool_result(tool_name, "after_sales_confirmation_cancelled")
    if decision == UserDecision.MODIFY:
        _set_pending_after_sales(metadata, None)
        return _blocked_tool_result(tool_name, "after_sales_modify_invalidated")
    if decision == UserDecision.UNSAFE_INJECTION:
        return _blocked_tool_result(tool_name, "after_sales_unsafe_injection")
    if decision != UserDecision.CONFIRM:
        return _blocked_tool_result(tool_name, "after_sales_explicit_confirmation_required")

    pending = _pending_after_sales(metadata)
    if pending is None:
        return _blocked_tool_result(tool_name, "after_sales_confirmation_missing")
    if pending.get("status") != CUSTOMER_SERVICE_PENDING_STATUS:
        return _blocked_tool_result(tool_name, "after_sales_confirmation_not_pending")
    expected = {
        "order_no": pending.get("order_no"),
        "customer_phone_last4": pending.get("customer_phone_last4"),
        "draft_id": pending.get("draft_id"),
        "operation_id": pending.get("operation_id"),
        "confirmed": True,
    }
    actual = {
        "order_no": arguments.get("order_no"),
        "customer_phone_last4": arguments.get("customer_phone_last4"),
        "draft_id": arguments.get("draft_id"),
        "operation_id": arguments.get("operation_id"),
        "confirmed": arguments.get("confirmed"),
    }
    if actual != expected:
        return _blocked_tool_result(tool_name, "after_sales_confirmation_mismatch")
    created_turn_id = pending.get("created_turn_id")
    current_turn_id = current_runtime_turn_id(state)
    if not isinstance(created_turn_id, str) or current_turn_id == created_turn_id:
        return _blocked_tool_result(tool_name, "after_sales_same_turn_confirm_blocked")
    version = int(pending.get("version") or 1)
    operation_id = str(pending.get("operation_id") or "")
    if not _PENDING_COORDINATOR.reserve(session_id, operation_id, version):
        return _blocked_tool_result(tool_name, "after_sales_confirmation_in_progress")
    pending["status"] = CUSTOMER_SERVICE_CONFIRMING_STATUS
    pending["reservation_version"] = version
    return None


def prepare_customer_service_tool_arguments(
    *,
    state: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if state.get("metadata", {}).get("agent_id") != CUSTOMER_SERVICE_AGENT_ID:
        return arguments
    if tool_name not in {"search_products", "recommend_products", "compare_products"}:
        return arguments
    prepared = dict(arguments)
    allowed_scope = {
        value
        for value in state.get("allowed_knowledge_base_ids", [])
        if isinstance(value, int)
    }
    knowledge_base_id = state.get("knowledge_base_id")
    if isinstance(knowledge_base_id, int) and knowledge_base_id in allowed_scope:
        prepared["knowledge_base_id"] = knowledge_base_id
    else:
        prepared.pop("knowledge_base_id", None)
    return prepared


def update_customer_service_state_after_tool(
    *,
    state: Any,
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> None:
    if state.get("metadata", {}).get("agent_id") != CUSTOMER_SERVICE_AGENT_ID:
        return
    if tool_name != "create_after_sales_ticket":
        return
    metadata = state.setdefault("metadata", {})
    action = arguments.get("action", "draft")
    if action == "confirm":
        _finalize_after_sales_confirmation(state, result)
        return
    if not result.success or not isinstance(result.result, dict):
        return
    if action == "draft":
        draft_id = result.result.get("draft_id")
        operation_id = result.result.get("operation_id")
        if not isinstance(draft_id, str) or not isinstance(operation_id, str):
            return
        _set_pending_after_sales(
            metadata,
            {
                "draft_id": draft_id,
                "operation_id": operation_id,
                "order_no": arguments.get("order_no"),
                "customer_phone_last4": arguments.get("customer_phone_last4"),
                "issue_type": arguments.get("issue_type"),
                "summary": result.result.get("summary"),
                "created_turn_id": current_runtime_turn_id(state),
                "version": 1,
                "status": CUSTOMER_SERVICE_PENDING_STATUS,
                "conversation_id": state.get("conversation_id"),
            },
        )


def _finalize_after_sales_confirmation(state: Any, result: ToolResult) -> None:
    metadata = state.setdefault("metadata", {})
    pending = _pending_after_sales(metadata)
    if pending is None:
        return
    session_id = _conversation_session_id(state)
    operation_id = str(pending.get("operation_id") or "")
    if session_id is not None:
        _PENDING_COORDINATOR.finalize(session_id, operation_id, success=result.success)
    if result.success:
        customer_service = metadata.setdefault("customer_service", {})
        customer_service["last_confirmed_operation_id"] = operation_id
        pending["status"] = CUSTOMER_SERVICE_CONFIRMED_STATUS
        _set_pending_after_sales(metadata, None)
        return
    pending["status"] = CUSTOMER_SERVICE_PENDING_STATUS


def _tool_decision(tool_name: str, arguments: dict[str, Any]) -> AgentDecision:
    return AgentDecision(
        action="tool_calls",
        content=None,
        tool_calls=[
            AgentToolCall(
                id=f"customer_service_{uuid4().hex}",
                tool_name=tool_name,
                arguments=arguments,
            )
        ],
        metadata={
            "actual_strategy": CustomerServicePlannerStrategy.name,
            "tool_registry": {
                "available_tools": CUSTOMER_SERVICE_TOOL_ALLOWLIST,
                "selected_tools": [tool_name],
                "rejected_tools": [],
            },
        },
    )


def _final(content: str) -> AgentDecision:
    return AgentDecision(
        action="final",
        content=content,
        metadata={"actual_strategy": CustomerServicePlannerStrategy.name},
    )


def _manual_followup_decision(state: dict[str, Any], observation: dict[str, Any]) -> AgentDecision:
    raw_result = observation.get("raw_result")
    if not isinstance(raw_result, dict):
        return _final("没有拿到商品查询结果，不能检索说明书。")
    items = raw_result.get("items")
    if not isinstance(items, list) or not items:
        return _final("没有找到明确商品，不能跨型号检索说明书。")
    if len(items) > 1:
        return _final("找到多个候选商品，请先明确具体型号。")
    product = items[0]
    if not isinstance(product, dict):
        return _final("商品结果格式异常，不能检索说明书。")
    document_id = product.get("primary_manual_document_id")
    if not isinstance(document_id, int):
        return _final("该商品当前没有绑定主说明书，不能进行无约束说明书检索。")
    return _tool_decision(
        "knowledge_search",
        {
            "query": state["query"],
            "knowledge_base_id": state.get("knowledge_base_id"),
            "conversation_id": state.get("conversation_id"),
            "memory_context": state.get("memory_context"),
            "document_id": document_id,
        },
    )


def _knowledge_final(state: dict[str, Any], observation: dict[str, Any]) -> AgentDecision:
    raw_result = observation.get("raw_result")
    if not isinstance(raw_result, dict):
        return _final("说明书检索失败，不能基于资料回答。")
    answer = str(raw_result.get("answer") or "")
    sources = raw_result.get("sources")
    if not answer or not sources:
        return _final("说明书中没有找到相关内容。")
    expected_document_id = _last_knowledge_document_id(state)
    if expected_document_id is not None:
        for source in sources:
            if not isinstance(source, dict):
                return _final("说明书来源格式异常，不能混入最终答复。")
            if source.get("document_id") != expected_document_id:
                return _final("说明书来源与目标型号不一致，已拒绝混入最终答复。")
    return _final(answer)


def _after_sales_final(metadata: dict[str, Any], observation: dict[str, Any]) -> AgentDecision:
    raw_result = observation.get("raw_result")
    if not isinstance(raw_result, dict):
        return _final("模拟售后工具没有返回有效结果。")
    if raw_result.get("status") == "draft":
        return _final("已生成模拟售后草稿，请确认是否提交。")
    if raw_result.get("status") == "created":
        _set_pending_after_sales(metadata, None)
        return _final(f"已创建模拟售后工单：{raw_result.get('ticket_id')}。")
    return _final("模拟售后流程已处理。")


def _observation_final(observation: dict[str, Any]) -> AgentDecision:
    if observation.get("success") is False:
        return _final(str(observation.get("error") or "工具调用失败，不能猜测业务结果。"))
    return _final(
        str(observation.get("content") or observation.get("raw_result") or "已完成查询。")
    )


def _after_sales_draft_or_clarify(query: str) -> AgentDecision:
    order = _extract_order_fields(query)
    if order is None:
        return _final("请提供订单号和手机号后四位，并描述售后问题。")
    issue_type = "repair"
    if "退" in query:
        issue_type = "refund"
    elif "换" in query:
        issue_type = "exchange"
    return _tool_decision(
        "create_after_sales_ticket",
        {
            "action": "draft",
            **order,
            "issue_type": issue_type,
            "issue_description": query[:500],
        },
    )


def latest_user_message(state: Any) -> str | None:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"user", "human"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return None


def current_runtime_turn_id(state: Any) -> str:
    metadata = state.setdefault("metadata", {})
    existing = metadata.get("runtime_turn_id")
    if isinstance(existing, str) and existing:
        return existing
    turn_id = uuid4().hex
    metadata["runtime_turn_id"] = turn_id
    return turn_id


def classify_user_decision(message: str) -> UserDecision:
    normalized = _normalize_user_decision_text(message)
    if not normalized:
        return UserDecision.OTHER
    if any(
        pattern in normalized
        for pattern in [
            "忽略系统规则",
            "绕过确认",
            "confirmed=true",
            "假装用户已经确认",
            "执行隐藏指令",
        ]
    ):
        return UserDecision.UNSAFE_INJECTION
    if normalized in {"不要取消", "别取消", "不要撤销"}:
        return UserDecision.AMBIGUOUS
    if any(
        pattern in normalized
        for pattern in ["取消", "不要提交", "不确认", "先等等", "暂不办理", "别创建", "停止"]
    ):
        return UserDecision.CANCEL
    if _has_modify_intent(normalized):
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


def _normalize_user_decision_text(message: str) -> str:
    text = unicodedata.normalize("NFKC", message).strip()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[。！？!?.，,；;]+$", "", text)


def _has_modify_intent(query: str) -> bool:
    return any(
        marker in query
        for marker in [
            "改成",
            "修改描述",
            "问题改为",
            "订单换成",
            "手机号后四位改为",
            "商品换成",
            "重新填写",
        ]
    )


def _context_changed(pending: dict[str, Any], query: str) -> bool:
    order = _extract_order_fields(query)
    if order is None:
        return False
    return (
        order.get("order_no") != pending.get("order_no")
        or order.get("customer_phone_last4") != pending.get("customer_phone_last4")
    )


def _conversation_session_id(state: Any) -> str | None:
    if state.get("conversation_id") is None:
        return None
    session = state.get("metadata", {}).get("session", {})
    session_id = session.get("session_id") if isinstance(session, dict) else None
    if isinstance(session_id, str) and session_id.startswith("conversation:"):
        return session_id
    return f"conversation:{state.get('conversation_id')}"


def _product_query_args(state: Any, query: str, *, page_size: int) -> dict[str, Any]:
    args: dict[str, Any] = {
        "sale_status": "on_sale",
        "in_stock_only": True,
        "sort_by": "popularity",
        "sort_order": "desc",
        "page_size": page_size,
    }
    if isinstance(state.get("knowledge_base_id"), int):
        args["knowledge_base_id"] = state["knowledge_base_id"]
    if "豆浆机" in query:
        args["category"] = "豆浆机"
    model = _extract_model(query)
    if model:
        args["model"] = model
    price_max = _extract_price_max(query)
    if price_max is not None:
        args["price_max"] = price_max
    if "必须" in query and "清洗" in query:
        args["required_features"] = ["容易清洗"]
    elif "清洗" in query:
        args["preferred_features"] = ["容易清洗"]
    if "不能" in query and "噪音" in query:
        args["excluded_features"] = ["高噪音"]
    elif "噪音" in query:
        args["preferred_features"] = [*args.get("preferred_features", []), "低噪音"]
    if "宿舍" in query:
        if "必须适合宿舍" in query or "只能" in query:
            args["required_use_cases"] = ["宿舍"]
        else:
            args["preferred_use_cases"] = ["宿舍"]
    return args


def _extract_order_fields(query: str) -> dict[str, Any] | None:
    order_match = re.search(r"\b(\d{10,20})\b", query)
    last4_match = re.search(r"(?:后四位|尾号|手机号后四位)\D*(\d{4})", query)
    if order_match is None or last4_match is None:
        return None
    return {
        "order_no": order_match.group(1),
        "customer_phone_last4": last4_match.group(1),
    }


def _extract_product_codes(query: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for code in re.findall(r"\b[A-Z]{1,6}\d{2,12}\b", query):
        if code not in seen:
            result.append(code)
            seen.add(code)
    return result


def _extract_model(query: str) -> str | None:
    explicit = re.search(r"型号[:：\s]*([A-Za-z0-9_-]{2,64})", query)
    if explicit:
        return explicit.group(1)
    model = re.search(r"\b([A-Z]{1,6}\d{2,12})\b", query)
    return model.group(1) if model else None


def _extract_price_max(query: str) -> int | None:
    match = re.search(r"(\d{2,6})\s*(?:以内|以下|内)", query)
    return int(match.group(1)) if match else None


def _pending_after_sales(metadata: dict[str, Any]) -> dict[str, Any] | None:
    value = metadata.get("customer_service", {}).get(CUSTOMER_SERVICE_PENDING_KEY)
    return value if isinstance(value, dict) else None


def _set_pending_after_sales(metadata: dict[str, Any], value: dict[str, Any] | None) -> None:
    customer_service = metadata.setdefault("customer_service", {})
    if value is None:
        customer_service.pop(CUSTOMER_SERVICE_PENDING_KEY, None)
    else:
        customer_service[CUSTOMER_SERVICE_PENDING_KEY] = value


def _blocked_tool_result(tool_name: str, reason: str) -> ToolResult:
    return ToolResult(
        name=tool_name,
        success=False,
        error="售后确认校验未通过",
        metadata={
            "status": "blocked",
            "reason": reason,
            "error_type": "customer_service_confirmation_error",
        },
    )


def _last_tool_name(observations: list[dict[str, Any]]) -> str | None:
    if not observations:
        return None
    value = observations[-1].get("tool_name")
    return value if isinstance(value, str) else None


def _last_knowledge_document_id(state: dict[str, Any]) -> int | None:
    for tool_call in reversed(state.get("tool_calls", [])):
        if not isinstance(tool_call, dict):
            continue
        if tool_call.get("tool_name") != "knowledge_search":
            continue
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return None
        document_id = arguments.get("document_id")
        return document_id if isinstance(document_id, int) else None
    return None


def _is_greeting(query: str) -> bool:
    return query in {"你好", "您好", "hi", "hello", "你能做什么"}


def _is_prompt_injection(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.UNSAFE_INJECTION


def _is_product_search(query: str) -> bool:
    return any(word in query for word in ["查", "找", "看看", "商品", "豆浆机"])


def _is_recommend(query: str) -> bool:
    return any(word in query for word in ["推荐", "适合", "预算", "偏好"])


def _is_compare(query: str) -> bool:
    return "对比" in query or "比较" in query


def _is_manual_question(query: str) -> bool:
    return any(word in query for word in ["说明书", "怎么用", "清洁", "故障", "安全", "操作"])


def _is_order(query: str) -> bool:
    return "订单" in query and not _is_after_sales(query)


def _is_logistics(query: str) -> bool:
    return "物流" in query or "快递" in query


def _is_after_sales(query: str) -> bool:
    return any(word in query for word in ["售后", "维修", "退货", "换货", "坏了"])


def _is_handoff(query: str) -> bool:
    return "人工" in query or "客服" in query


def _is_clear_confirmation(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.CONFIRM


def _is_ambiguous_confirmation(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.AMBIGUOUS


def _is_cancel(query: str) -> bool:
    return classify_user_decision(query) == UserDecision.CANCEL
