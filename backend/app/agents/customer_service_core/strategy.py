from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from backend.app.agents.customer_service import CustomerServiceHybridPlannerStrategy
from backend.app.agents.customer_service_core.adapters import (
    CommandAdapter,
    command_from_tool_call,
)
from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceExecution,
    CustomerServiceState,
    ExecutionPhase,
    GoalSnapshot,
    KnowledgeSearchCommand,
    PendingTransaction,
    TransactionStatus,
)
from backend.app.agents.customer_service_core.dst import (
    load_dst,
    record_candidate_batch,
    replace_domain_candidates,
)
from backend.app.agents.customer_service_core.schemas import CustomerServiceDomain
from backend.app.agents.langgraph.tool_calling import (
    AgentDecision,
    AgentToolCall,
    BaseAgentPlannerStrategy,
)
from pydantic import ValidationError


class CustomerServiceStrategy(BaseAgentPlannerStrategy):
    """唯一正式客服策略。

    现阶段复用已验证的安全/语义理解能力，但所有业务调用必须转换为
    强类型 Command，并由显式执行阶段控制后续轮次。
    """

    name = "customer_service"

    async def adecide(self, state: Any) -> AgentDecision:
        execution = self._load_execution(state)
        if execution.phase == ExecutionPhase.FAILED:
            return AgentDecision(
                action="final",
                content="本次业务操作未成功，可信状态未发生变化。",
                metadata={"actual_strategy": self.name, "execution_phase": execution.phase},
            )
        if execution.phase == ExecutionPhase.READY_FOR_FINAL:
            return AgentDecision(
                action="final",
                content=None,
                metadata={"actual_strategy": self.name, "execution_phase": execution.phase},
            )
        if execution.tool_count >= 2:
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = "customer_service_tool_limit"
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(
                action="final",
                content="本轮需要的业务校验步骤过多，请缩小查询范围后重试。",
                metadata={"actual_strategy": self.name, "execution_phase": execution.phase},
            )

        first_pass = execution.phase == ExecutionPhase.NEW
        if first_pass:
            execution.goal = GoalSnapshot(raw_query=str(state.get("query") or ""))
        elif execution.phase != ExecutionPhase.CONTINUE:
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = "invalid_execution_phase"
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(action="final", content="客服执行阶段异常，本轮未修改状态。")

        if execution.phase == ExecutionPhase.CONTINUE:
            decision = self._continue_after_verification(state, execution)
        else:
            planning_state = self._planning_state(state)
            decision = await CustomerServiceHybridPlannerStrategy().adecide(planning_state)
            self._capture_understanding(planning_state, execution)
        decision.metadata.update(
            {
                "actual_strategy": self.name,
                "understanding_reused": not first_pass,
            }
        )
        if not decision.tool_calls:
            execution.phase = (
                ExecutionPhase.READY_FOR_CLARIFICATION
                if decision.content and "请" in decision.content
                else ExecutionPhase.READY_FOR_FINAL
            )
            state["customer_service_execution"] = execution.model_dump(mode="json")
            self._record_details(state, decision, execution)
            return decision

        if len(decision.tool_calls) != 1:
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = "parallel_business_tools_forbidden"
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(
                action="final",
                content="本轮业务请求包含多个并行动作，请一次处理一个目标。",
                metadata={"actual_strategy": self.name},
            )

        tool_call = decision.tool_calls[0]
        try:
            command = command_from_tool_call(tool_call.tool_name, tool_call.arguments)
            tool_name, arguments = CommandAdapter().adapt(command)
        except (TypeError, ValueError, ValidationError) as exc:
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = f"command_schema_failed:{exc}"
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(
                action="final",
                content="业务参数校验未通过，本轮未执行工具。",
                metadata={"actual_strategy": self.name, "command_error": str(exc)},
            )

        tool_call = tool_call.model_copy(
            update={"tool_name": tool_name, "arguments": arguments}
        )
        transaction = PendingTransaction(
            transaction_id=f"cs_tx_{uuid4().hex}",
            turn_id=execution.turn_id,
            sequence=execution.tool_count + 1,
            command=command,
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            arguments_hash=self._arguments_hash(arguments),
            proposed_patch=self._proposed_patch(command),
            expected_result_type=self._expected_result_type(tool_name, arguments, state),
            status=TransactionStatus.EXECUTING,
        )
        execution.pending_transaction = transaction
        execution.phase = ExecutionPhase.WAITING_TOOL
        state["customer_service_execution"] = execution.model_dump(mode="json")
        decision.tool_calls = [tool_call]
        self._record_details(state, decision, execution)
        return decision

    @staticmethod
    def _planning_state(state: dict[str, Any]) -> dict[str, Any]:
        """把唯一正式状态投影成只在 Planner 调用期间存在的兼容视图。"""
        planning_state = dict(state)
        source_metadata = state.get("metadata", {})
        planning_metadata = dict(source_metadata)
        planning_metadata["customer_service"] = deepcopy(
            source_metadata.get("customer_service", {})
        )
        planning_state["metadata"] = planning_metadata
        customer_service = planning_state["metadata"].setdefault(
            "customer_service",
            {},
        )
        formal = CustomerServiceState.model_validate(customer_service.get("state", {}))
        preserved_pending = customer_service.get("pending_after_sales")
        customer_service.clear()
        customer_service["state"] = formal.model_dump(mode="json")
        if preserved_pending is not None:
            customer_service["pending_after_sales"] = preserved_pending
        dst = load_dst(planning_state["metadata"])
        product_items = (
            formal.candidate_batches[-1].items
            if formal.candidate_batches
            else []
        )
        replace_domain_candidates(
            dst,
            domain=CustomerServiceDomain.PRODUCT,
            candidates=[
                {
                    "ref": item.product_code,
                    "display_name": item.name,
                    "category": item.category,
                    "primary_manual_document_id": item.primary_manual_document_id,
                }
                for item in product_items
            ],
            active_ref=formal.active_product_code,
            filters=formal.filters,
        )
        for batch in formal.candidate_batches:
            record_candidate_batch(
                dst,
                domain=CustomerServiceDomain.PRODUCT,
                batch_id=batch.batch_id,
                candidates=[
                    {
                        "ref": item.product_code,
                        "display_name": item.name,
                        "category": item.category,
                        "primary_manual_document_id": item.primary_manual_document_id,
                    }
                    for item in batch.items
                ],
            )
        replace_domain_candidates(
            dst,
            domain=CustomerServiceDomain.ORDER,
            candidates=[
                {
                    "ref": item.get("order_ref") or item.get("order_no"),
                    "display_name": item.get("product_name"),
                    **item,
                }
                for item in formal.order_candidates
                if item.get("order_ref") or item.get("order_no")
            ],
            active_ref=formal.active_order_ref,
        )
        customer_service["dst"] = dst.model_dump(mode="json")
        customer_service["product_filters"] = formal.filters
        customer_service["order_candidates"] = formal.order_candidates
        if formal.active_order_ref:
            customer_service["active_order_ref"] = formal.active_order_ref
        return planning_state

    @staticmethod
    def _capture_understanding(
        planning_state: dict[str, Any],
        execution: CustomerServiceExecution,
    ) -> None:
        customer_service = planning_state.get("metadata", {}).get(
            "customer_service",
            {},
        )
        route = customer_service.get("route")
        if execution.goal is not None and isinstance(route, dict):
            intent = route.get("intent")
            execution.goal.intent = str(intent) if intent is not None else None

    def _continue_after_verification(
        self,
        state: dict[str, Any],
        execution: CustomerServiceExecution,
    ) -> AgentDecision:
        business_state = CustomerServiceState.model_validate(
            state.get("metadata", {})
            .get("customer_service", {})
            .get("state", {})
        )
        latest = (
            business_state.candidate_batches[-1]
            if business_state.candidate_batches
            else None
        )
        if latest is None or not latest.items:
            execution.phase = ExecutionPhase.READY_FOR_FINAL
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(action="final", content="没有找到该商品，无法查询说明书。")
        if len(latest.items) > 1:
            execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
            state["customer_service_execution"] = execution.model_dump(mode="json")
            names = "、".join(item.name for item in latest.items)
            return AgentDecision(
                action="final",
                content=f"找到多个匹配商品：{names}。请明确选择一个。",
            )
        product = latest.items[0]
        if product.primary_manual_document_id is None:
            execution.phase = ExecutionPhase.READY_FOR_FINAL
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(action="final", content="该商品没有绑定可用的主说明书。")
        allowed = {
            value
            for value in state.get("allowed_knowledge_base_ids", [])
            if isinstance(value, int)
        }
        knowledge_base_id = state.get("knowledge_base_id")
        if not isinstance(knowledge_base_id, int) or knowledge_base_id not in allowed:
            execution.phase = ExecutionPhase.FAILED
            execution.failure_reason = "knowledge_base_scope_missing"
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return AgentDecision(
                action="final",
                content="当前会话没有可用的说明书知识库权限。",
            )
        command = KnowledgeSearchCommand(
            query=execution.goal.raw_query if execution.goal else str(state.get("query") or ""),
            knowledge_base_id=knowledge_base_id,
            document_id=product.primary_manual_document_id,
            conversation_id=state.get("conversation_id"),
            memory_context=state.get("memory_context"),
        )
        _, arguments = CommandAdapter().adapt(command)
        return AgentDecision(
            action="tool_calls",
            tool_calls=[
                AgentToolCall(
                    id=f"cs_tool_{uuid4().hex}",
                    tool_name=command.kind,
                    arguments=arguments,
                )
            ],
        )

    @staticmethod
    def _load_execution(state: dict[str, Any]) -> CustomerServiceExecution:
        raw = state.get("customer_service_execution")
        if isinstance(raw, dict):
            return CustomerServiceExecution.model_validate(raw)
        turn_id = str(
            state.get("metadata", {}).get("_runtime_turn_id")
            or state.get("metadata", {}).get("runtime_turn_id")
            or f"turn_{uuid4().hex}"
        )
        execution = CustomerServiceExecution(turn_id=turn_id)
        state["customer_service_execution"] = execution.model_dump(mode="json")
        return execution

    @staticmethod
    def _arguments_hash(arguments: dict[str, Any]) -> str:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _proposed_patch(command: Any) -> dict[str, Any]:
        filters = getattr(command, "filters", None)
        return {"filters": filters} if isinstance(filters, dict) else {}

    @staticmethod
    def _expected_result_type(
        tool_name: str,
        arguments: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        if (
            tool_name == "search_products"
            and arguments.get("product_code")
            and any(
                token in str(state.get("query") or "")
                for token in ("说明书", "怎么用", "按键", "蓝牙", "连接")
            )
        ):
            return "product_verification"
        return {
            "search_products": "products",
            "recommend_products": "products",
            "compare_products": "products",
            "knowledge_search": "knowledge",
            "query_order": "order",
            "query_logistics": "logistics",
            "create_after_sales_ticket": "after_sales",
            "create_human_handoff": "handoff",
        }.get(tool_name, "unknown")

    @staticmethod
    def _record_details(
        state: dict[str, Any],
        decision: AgentDecision,
        execution: CustomerServiceExecution,
    ) -> None:
        details = state.setdefault("metadata", {}).setdefault(
            "customer_service",
            {},
        ).setdefault("execution_details", {})
        details.setdefault("safety", {"status": "passed"})
        details.setdefault("understanding", execution.goal.model_dump() if execution.goal else {})
        details.setdefault("merge", {})
        details.setdefault(
            "state_before",
            state.get("metadata", {}).get("customer_service", {}).get("state", {}),
        )
        details.setdefault(
            "state_preview",
            (
                execution.pending_transaction.proposed_patch
                if execution.pending_transaction
                else {}
            ),
        )
        details.setdefault("resolution", {})
        details["command"] = (
            execution.pending_transaction.command.model_dump(mode="json")
            if execution.pending_transaction
            else None
        )
        details["tool_args"] = (
            decision.tool_calls[0].arguments if decision.tool_calls else None
        )
