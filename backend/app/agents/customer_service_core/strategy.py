from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from backend.app.agents.customer_service_core.adapters import CommandAdapter
from backend.app.agents.customer_service_core.command_builder import build_command
from backend.app.agents.customer_service_core.commit import CommitCoordinator
from backend.app.agents.customer_service_core.contracts import (
    CustomerServiceCommand,
    CustomerServiceExecution,
    CustomerServiceState,
    ExecutionPhase,
    GoalSnapshot,
    KnowledgeSearchCommand,
    PendingProductQuery,
    PendingTransaction,
    ProductContext,
    StatePreview,
    TransactionStatus,
)
from backend.app.agents.customer_service_core.read_only_fallback import (
    recommend_read_only_capability,
)
from backend.app.agents.customer_service_core.reducer import reduce_state
from backend.app.agents.customer_service_core.understanding import understand
from backend.app.agents.langgraph.tool_calling import (
    AgentDecision,
    AgentToolCall,
    BaseAgentPlannerStrategy,
)
from pydantic import ValidationError


class CustomerServiceStrategy(BaseAgentPlannerStrategy):
    name = "customer_service"

    async def adecide(self, state: Any) -> AgentDecision:
        execution = self._load_execution(state)
        if execution.phase == ExecutionPhase.FAILED:
            return self._final("本次业务操作未成功，可信状态未发生变化。")
        if execution.phase == ExecutionPhase.READY_FOR_FINAL:
            return self._final()
        if execution.phase == ExecutionPhase.CONTINUE:
            return self._continue(state, execution)
        if execution.phase != ExecutionPhase.NEW:
            return self._fail(state, execution, "invalid_execution_phase")
        if execution.tool_count >= 2:
            return self._fail(state, execution, "customer_service_tool_limit")

        business_state = self._business_state(state)
        details = self._reset_details(state, business_state)
        understanding = await understand(
            query=str(state.get("query") or ""),
            state=business_state,
            messages=state.get("messages", []),
        )
        execution.goal = GoalSnapshot(
            raw_query=str(state.get("query") or ""),
            intent=understanding.frame.intent,
            semantic_frame=understanding.frame,
        )
        details["safety"] = {
            "status": ("blocked" if understanding.frame.intent == "blocked" else "passed")
        }
        details["understanding"] = understanding.model_dump(mode="json")
        details["merge"] = understanding.merge

        preview = reduce_state(business_state, understanding.frame)
        details["state_preview"] = preview.model_dump(mode="json")
        build = build_command(
            frame=understanding.frame,
            preview=preview,
            runtime=self._runtime_scope(state),
        )
        if (
            understanding.llm_used
            and understanding.frame.intent
            in {
                "other",
                "recommend_products",
                "search_products",
                "compare_products",
                "order",
                "logistics",
                "product_fact",
            }
            and build.clarification is not None
        ):
            fallback = await recommend_read_only_capability(
                rewritten_query=understanding.rewritten_query or execution.goal.raw_query,
                state=business_state,
            )
            details["read_only_tool_fallback"] = fallback.model_dump(mode="json")
            if fallback.semantic_frame is not None:
                fallback_preview = reduce_state(
                    business_state,
                    fallback.semantic_frame,
                )
                fallback_build = build_command(
                    frame=fallback.semantic_frame,
                    preview=fallback_preview,
                    runtime=self._runtime_scope(state),
                )
                if fallback_build.command is not None:
                    execution.goal.intent = fallback.semantic_frame.intent
                    execution.goal.semantic_frame = fallback.semantic_frame
                    preview = fallback_preview
                    build = fallback_build
                    details["state_preview"] = preview.model_dump(mode="json")
        details["resolution"] = build.resolution.model_dump(mode="json") if build.resolution else {}
        execution.goal.resolved_entities = (
            build.resolution.product_codes if build.resolution else []
        )
        if build.direct_answer is not None:
            if build.state_action == "set_pending_product_query":
                category = understanding.frame.slots.get("category")
                keyword = understanding.frame.slots.get("keyword")
                if not isinstance(category, str) or not isinstance(keyword, str):
                    return self._fail(
                        state,
                        execution,
                        "pending_product_query_contract_failed",
                    )
                preview.proposed_state.product = ProductContext(
                    pending_query=PendingProductQuery(
                        category=category,
                        keyword=keyword,
                        requested_count=understanding.frame.requested_count or 1,
                        created_turn_id=execution.turn_id,
                    )
                )
                CommitCoordinator().commit_preview(
                    agent_state=state,
                    state=preview.proposed_state,
                )
                details["state_commit"] = preview.proposed_state.model_dump(mode="json")
            elif understanding.frame.intent == "cancel":
                CommitCoordinator().commit_preview(
                    agent_state=state,
                    state=preview.proposed_state,
                )
                details["state_commit"] = preview.proposed_state.model_dump(mode="json")
            execution.phase = ExecutionPhase.READY_FOR_FINAL
            state["customer_service_execution"] = execution.model_dump(mode="json")
            details["command"] = None
            return self._final(build.direct_answer)
        if build.clarification is not None:
            execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
            state["customer_service_execution"] = execution.model_dump(mode="json")
            details["command"] = None
            return self._final(build.clarification)
        if build.command is None:
            return self._fail(state, execution, "command_missing")
        if preview.proposed_patch.get("invalidate_product_context") is True:
            invalidated = business_state.model_copy(deep=True)
            invalidated.product = ProductContext()
            CommitCoordinator().commit_preview(
                agent_state=state,
                state=invalidated,
            )
            details["state_invalidation"] = invalidated.model_dump(mode="json")
        return self._tool_decision(
            state=state,
            execution=execution,
            command=build.command,
            preview=preview,
            expected_result_type=build.expected_result_type,
        )

    def _continue(
        self,
        state: dict[str, Any],
        execution: CustomerServiceExecution,
    ) -> AgentDecision:
        if execution.tool_count >= 2:
            return self._fail(state, execution, "customer_service_tool_limit")
        transaction = execution.pending_transaction
        if transaction is None:
            return self._fail(state, execution, "pending_transaction_missing")
        expected = transaction.expected_result_type
        business_state = self._business_state(state)
        if expected in {
            "product_verification_for_manual",
            "product_selection_for_manual_fact",
        }:
            verified = (
                execution.verified_products
                if expected == "product_verification_for_manual"
                else (
                    business_state.product.active_batch.items
                    if business_state.product.active_batch is not None
                    else []
                )
            )
            if not verified:
                execution.phase = ExecutionPhase.READY_FOR_FINAL
                state["customer_service_execution"] = execution.model_dump(mode="json")
                return self._final("没有找到该商品，无法查询说明书。")
            if len(verified) != 1:
                execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
                state["customer_service_execution"] = execution.model_dump(mode="json")
                names = "、".join(item.name for item in verified)
                return self._final(f"找到多个匹配商品：{names}。请明确选择一个。")
            product = verified[0]
            if product.primary_manual_document_id is None:
                execution.phase = ExecutionPhase.READY_FOR_FINAL
                state["customer_service_execution"] = execution.model_dump(mode="json")
                return self._final("该商品没有绑定可用的主说明书。")
            knowledge_base_id = self._allowed_knowledge_base_id(state)
            if knowledge_base_id is None:
                return self._fail(state, execution, "knowledge_base_scope_missing")
            command = KnowledgeSearchCommand(
                query=execution.goal.raw_query if execution.goal else "",
                knowledge_base_id=knowledge_base_id,
                document_id=product.primary_manual_document_id,
                conversation_id=state.get("conversation_id"),
                memory_context=state.get("memory_context"),
            )
            preview = StatePreview(
                state_before=business_state,
                proposed_state=business_state.model_copy(deep=True),
            )
            return self._tool_decision(
                state=state,
                execution=execution,
                command=command,
                preview=preview,
                expected_result_type="knowledge",
            )
        if expected in {"order_list_for_logistics", "order_list_for_after_sales"}:
            execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
            state["customer_service_execution"] = execution.model_dump(mode="json")
            return self._final("请从刚才列出的订单中选择一笔继续处理。")
        if expected.startswith("order_verification_for_"):
            if execution.verified_order_ref is None or execution.goal is None:
                return self._fail(state, execution, "verified_order_missing")
            frame = execution.goal.semantic_frame
            if frame is None:
                return self._fail(state, execution, "semantic_frame_missing")
            preview = StatePreview(
                state_before=business_state,
                proposed_state=business_state.model_copy(deep=True),
            )
            build = build_command(
                frame=frame,
                preview=preview,
                runtime=self._runtime_scope(state, execution),
            )
            if build.command is None:
                return self._fail(state, execution, "verified_order_command_missing")
            return self._tool_decision(
                state=state,
                execution=execution,
                command=build.command,
                preview=preview,
                expected_result_type=build.expected_result_type,
            )
        execution.phase = ExecutionPhase.READY_FOR_FINAL
        state["customer_service_execution"] = execution.model_dump(mode="json")
        return self._final()

    def _tool_decision(
        self,
        *,
        state: dict[str, Any],
        execution: CustomerServiceExecution,
        command: CustomerServiceCommand,
        preview: StatePreview,
        expected_result_type: str,
    ) -> AgentDecision:
        try:
            tool_name, arguments = CommandAdapter().adapt(command)
        except (TypeError, ValueError, ValidationError) as exc:
            return self._fail(state, execution, f"command_schema_failed:{exc}")
        tool_call_id = f"customer_service_{uuid4().hex}"
        transaction = PendingTransaction(
            transaction_id=f"cs_tx_{uuid4().hex}",
            turn_id=execution.turn_id,
            sequence=execution.tool_count + 1,
            command=command,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments_hash=self._arguments_hash(arguments),
            proposed_patch=preview.proposed_patch,
            expected_result_type=expected_result_type,
            status=TransactionStatus.EXECUTING,
        )
        execution.pending_transaction = transaction
        execution.phase = ExecutionPhase.WAITING_TOOL
        state["customer_service_execution"] = execution.model_dump(mode="json")
        details = self._details(state)
        details["command"] = command.model_dump(mode="json")
        details["tool_args"] = arguments
        return AgentDecision(
            action="tool_calls",
            tool_calls=[
                AgentToolCall(
                    id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            ],
            metadata={
                "actual_strategy": self.name,
                "execution_phase": execution.phase,
            },
        )

    @staticmethod
    def _business_state(state: dict[str, Any]) -> CustomerServiceState:
        return CustomerServiceState.model_validate(
            state.get("metadata", {}).get("customer_service", {}).get("state", {})
        )

    @staticmethod
    def _load_execution(state: dict[str, Any]) -> CustomerServiceExecution:
        raw = state.get("customer_service_execution")
        if isinstance(raw, dict):
            return CustomerServiceExecution.model_validate(raw)
        turn_id = str(state.get("metadata", {}).get("runtime_turn_id") or f"turn_{uuid4().hex}")
        execution = CustomerServiceExecution(turn_id=turn_id)
        state["customer_service_execution"] = execution.model_dump(mode="json")
        return execution

    @staticmethod
    def _runtime_scope(
        state: dict[str, Any],
        execution: CustomerServiceExecution | None = None,
    ) -> dict[str, Any]:
        return {
            "conversation_id": state.get("conversation_id"),
            "knowledge_base_id": state.get("knowledge_base_id"),
            "allowed_knowledge_base_ids": state.get("allowed_knowledge_base_ids", []),
            "verified_order_ref": (execution.verified_order_ref if execution is not None else None),
        }

    @staticmethod
    def _allowed_knowledge_base_id(state: dict[str, Any]) -> int | None:
        knowledge_base_id = state.get("knowledge_base_id")
        allowed = {
            value for value in state.get("allowed_knowledge_base_ids", []) if isinstance(value, int)
        }
        return (
            knowledge_base_id
            if isinstance(knowledge_base_id, int) and knowledge_base_id in allowed
            else None
        )

    @staticmethod
    def _arguments_hash(arguments: dict[str, Any]) -> str:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _reset_details(
        state: dict[str, Any],
        business_state: CustomerServiceState,
    ) -> dict[str, Any]:
        customer_service = state.setdefault("metadata", {}).setdefault(
            "customer_service",
            {},
        )
        details: dict[str, Any] = {
            "safety": {},
            "understanding": {},
            "merge": {},
            "state_before": business_state.model_dump(mode="json"),
            "state_preview": {},
            "resolution": {},
            "command": None,
            "tool_args": None,
            "tool_result": None,
            "state_commit": None,
            "final_evidence": None,
            "final_answer": None,
        }
        customer_service["execution_details"] = details
        return details

    @staticmethod
    def _details(state: dict[str, Any]) -> dict[str, Any]:
        return (
            state.setdefault("metadata", {})
            .setdefault("customer_service", {})
            .setdefault("execution_details", {})
        )

    def _fail(
        self,
        state: dict[str, Any],
        execution: CustomerServiceExecution,
        reason: str,
    ) -> AgentDecision:
        execution.phase = ExecutionPhase.FAILED
        execution.failure_reason = reason
        state["customer_service_execution"] = execution.model_dump(mode="json")
        return self._final("本次业务操作未成功，可信状态未发生变化。")

    def _final(self, content: str | None = None) -> AgentDecision:
        return AgentDecision(
            action="final",
            content=content,
            metadata={"actual_strategy": self.name},
        )
