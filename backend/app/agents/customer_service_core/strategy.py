from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from backend.app.agents.customer_service_core.adapters import CommandAdapter
from backend.app.agents.customer_service_core.capability_selector import (
    select_capability,
)
from backend.app.agents.customer_service_core.command_builder import build_command
from backend.app.agents.customer_service_core.commit import CommitCoordinator
from backend.app.agents.customer_service_core.context_lifecycle import (
    reconcile_visible_context,
)
from backend.app.agents.customer_service_core.contracts import (
    ClarificationKind,
    CustomerServiceCommand,
    CustomerServiceExecution,
    CustomerServiceState,
    DialogDomain,
    DialogFocus,
    ExecutionPhase,
    GoalSnapshot,
    KnowledgeSearchCommand,
    PendingClarification,
    PendingProductQuery,
    PendingTransaction,
    ProductContext,
    ResumeAction,
    SemanticFrame,
    StatePreview,
    TransactionStatus,
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
        reconciled_state, lifecycle_changes = reconcile_visible_context(
            business_state,
            state.get("messages", []),
        )
        if lifecycle_changes:
            self._save_business_state(state, reconciled_state)
            business_state = reconciled_state
            details["context_lifecycle"] = {
                "changes": lifecycle_changes,
                "state_after": reconciled_state.model_dump(mode="json"),
            }
        understanding = await understand(
            query=str(state.get("query") or ""),
            state=business_state,
            messages=state.get("messages", []),
        )
        understanding.frame = self._merge_pending_safe_slots(
            business_state,
            understanding.frame,
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
        details["llm_stages"] = {
            name: detail.model_dump(mode="json")
            for name, detail in understanding.stage_details.items()
        }
        details["llm_call_count"] = understanding.llm_call_count

        preview = reduce_state(business_state, understanding.frame)
        details["state_preview"] = preview.model_dump(mode="json")
        build = build_command(
            frame=understanding.frame,
            preview=preview,
            runtime=self._runtime_scope(state),
        )
        if (
            understanding.capability_selector_required
            or (
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
            )
        ):
            fallback = await select_capability(
                rewritten_query=understanding.rewritten_query or execution.goal.raw_query,
                state=business_state,
            )
            details["capability_selection"] = fallback.model_dump(mode="json")
            details["llm_stages"].update(
                {
                    name: detail.model_dump(mode="json")
                    for name, detail in fallback.stage_details.items()
                }
            )
            details["llm_call_count"] += fallback.llm_call_count
            if fallback.semantic_frame is not None:
                fallback.semantic_frame = self._merge_pending_safe_slots(
                    business_state,
                    fallback.semantic_frame,
                )
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
            elif build.state_action == "rearm_after_sales_confirmation":
                rearmed = business_state.model_copy(deep=True)
                rearmed.dialog_focus = DialogFocus(
                    active_domain="after_sales",
                    active_action="awaiting_explicit_confirmation",
                    source_turn_id=execution.turn_id,
                )
                self._save_business_state(state, rearmed)
                details["state_commit"] = rearmed.model_dump(mode="json")
            execution.phase = ExecutionPhase.READY_FOR_FINAL
            state["customer_service_execution"] = execution.model_dump(mode="json")
            details["command"] = None
            return self._final(build.direct_answer)
        if build.clarification is not None:
            execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
            state["customer_service_execution"] = execution.model_dump(mode="json")
            details["command"] = None
            frame = execution.goal.semantic_frame or understanding.frame
            answer = self._record_or_escalate_clarification(
                state=state,
                business_state=business_state,
                execution=execution,
                frame=frame,
                clarification=build.clarification,
                ambiguous=build.resolution is not None,
            )
            return self._final(answer)
        if build.command is None:
            return self._fail(state, execution, "command_missing")
        if (
            business_state.pending_clarification is not None
            and not self._requires_continuation(build.expected_result_type)
        ):
            preview.proposed_state.pending_clarification = None
            preview.proposed_patch["pending_clarification"] = None
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
                execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
                state["customer_service_execution"] = execution.model_dump(mode="json")
                frame = execution.goal.semantic_frame if execution.goal is not None else None
                if frame is None:
                    return self._fail(state, execution, "semantic_frame_missing")
                return self._final(
                    self._record_or_escalate_clarification(
                        state=state,
                        business_state=business_state,
                        execution=execution,
                        frame=frame,
                        clarification="没有找到该商品，暂时无法查询对应说明书。",
                        ambiguous=False,
                        clarification_kind="missing_evidence",
                    )
                )
            if len(verified) != 1:
                execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
                state["customer_service_execution"] = execution.model_dump(mode="json")
                names = "、".join(item.name for item in verified)
                return self._final(f"找到多个匹配商品：{names}。请明确选择一个。")
            product = verified[0]
            if product.primary_manual_document_id is None:
                execution.phase = ExecutionPhase.READY_FOR_CLARIFICATION
                state["customer_service_execution"] = execution.model_dump(mode="json")
                frame = execution.goal.semantic_frame if execution.goal is not None else None
                if frame is None:
                    return self._fail(state, execution, "semantic_frame_missing")
                return self._final(
                    self._record_or_escalate_clarification(
                        state=state,
                        business_state=business_state,
                        execution=execution,
                        frame=frame,
                        clarification="该商品没有绑定可用的主说明书，暂时无法可靠回答。",
                        ambiguous=False,
                        clarification_kind="missing_evidence",
                    )
                )
            knowledge_base_id = self._allowed_knowledge_base_id(state)
            if knowledge_base_id is None:
                return self._fail(state, execution, "knowledge_base_scope_missing")
            command = KnowledgeSearchCommand(
                query=self._knowledge_query(execution, product.name),
                knowledge_base_id=knowledge_base_id,
                document_id=product.primary_manual_document_id,
                conversation_id=state.get("conversation_id"),
                memory_context=state.get("memory_context"),
            )
            preview = StatePreview(
                state_before=business_state,
                proposed_state=business_state.model_copy(deep=True),
            )
            if business_state.pending_clarification is not None:
                preview.proposed_state.pending_clarification = None
                preview.proposed_patch["pending_clarification"] = None
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
            if (
                business_state.pending_clarification is not None
                and not self._requires_continuation(build.expected_result_type)
            ):
                preview.proposed_state.pending_clarification = None
                preview.proposed_patch["pending_clarification"] = None
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

    @staticmethod
    def _knowledge_query(
        execution: CustomerServiceExecution,
        product_name: str,
    ) -> str:
        frame = execution.goal.semantic_frame if execution.goal is not None else None
        question = frame.question if frame is not None else None
        if not isinstance(question, str) or not question.strip():
            question = execution.goal.raw_query if execution.goal is not None else ""
        return f"{product_name}：{question.strip()}"

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
    def _save_business_state(
        state: dict[str, Any],
        business_state: CustomerServiceState,
    ) -> None:
        state.setdefault("metadata", {}).setdefault("customer_service", {})[
            "state"
        ] = business_state.model_dump(mode="json")

    def _record_or_escalate_clarification(
        self,
        *,
        state: dict[str, Any],
        business_state: CustomerServiceState,
        execution: CustomerServiceExecution,
        frame: SemanticFrame,
        clarification: str,
        ambiguous: bool,
        clarification_kind: ClarificationKind | None = None,
    ) -> str:
        domain = self._domain_for_intent(frame.intent)
        kind = clarification_kind or (
            "ambiguous_reference" if ambiguous else "missing_slot"
        )
        target = f"{domain}:{frame.intent}:{clarification}"
        existing = business_state.pending_clarification
        if (
            existing is not None
            and existing.domain == domain
            and existing.kind == kind
            and existing.target_description == target
        ):
            business_state.pending_clarification = None
            self._save_business_state(state, business_state)
            self._details(state)["clarification"] = {
                "status": "escalated",
                "attempts": 2,
                "target": target,
            }
            return "连续两次仍无法确认您的具体需求，建议转人工客服继续处理。"

        pending = PendingClarification(
            clarification_id=f"clarification_{uuid4().hex}",
            kind=kind,
            domain=domain,
            target_description=target,
            missing_fields=self._missing_fields(frame),
            resume_action=ResumeAction(
                action=frame.intent,
                domain=domain,
                safe_slots=self._safe_resume_slots(frame),
            ),
            attempts=1,
            created_turn_id=execution.turn_id,
            last_asked_turn_id=execution.turn_id,
        )
        business_state.pending_clarification = pending
        self._save_business_state(state, business_state)
        self._details(state)["clarification"] = {
            "status": "waiting",
            "pending": pending.model_dump(mode="json"),
        }
        return clarification

    @staticmethod
    def _domain_for_intent(intent: str) -> DialogDomain:
        if intent in {
            "recommend_products",
            "search_products",
            "compare_products",
            "product_catalog_detail",
        }:
            return "product"
        if intent in {"product_fact", "product_fact_with_selection"}:
            return "manual"
        if intent == "order":
            return "order"
        if intent == "logistics":
            return "logistics"
        if intent == "after_sales":
            return "after_sales"
        if intent == "handoff":
            return "handoff"
        return "general"

    @staticmethod
    def _missing_fields(frame: SemanticFrame) -> list[str]:
        if frame.intent == "after_sales":
            required = ("order_ref", "phone_last4", "issue_description")
        elif frame.intent == "handoff":
            required = ("order_ref", "phone_last4")
        elif frame.intent in {"recommend_products", "search_products"}:
            required = ("keyword",)
        else:
            return ["target"]
        return [field for field in required if not frame.slots.get(field)]

    @staticmethod
    def _safe_resume_slots(frame: SemanticFrame) -> dict[str, Any]:
        allowed = {
            "keyword",
            "category",
            "brand",
            "model",
            "order_ref",
            "phone_last4",
            "issue_type",
            "issue_description",
            "message",
        }
        return {
            key: value
            for key, value in frame.slots.items()
            if key in allowed and value is not None
        }

    def _merge_pending_safe_slots(
        self,
        state: CustomerServiceState,
        frame: SemanticFrame,
    ) -> SemanticFrame:
        pending = state.pending_clarification
        resume = pending.resume_action if pending is not None else None
        if resume is None or resume.action != frame.intent:
            return frame
        if resume.domain != self._domain_for_intent(frame.intent):
            return frame
        merged = frame.model_copy(deep=True)
        merged.slots = {**resume.safe_slots, **frame.slots}
        return merged

    @staticmethod
    def _requires_continuation(expected_result_type: str) -> bool:
        return expected_result_type in {
            "product_verification_for_manual",
            "product_selection_for_manual_fact",
            "order_list_for_logistics",
            "order_list_for_after_sales",
        } or expected_result_type.startswith("order_verification_for_")

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
            "llm_stages": {},
            "llm_call_count": 0,
            "llm_calls_accounted": False,
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
