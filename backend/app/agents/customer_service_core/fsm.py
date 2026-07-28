from __future__ import annotations

from enum import StrEnum

from backend.app.agents.customer_service_core.schemas import (
    AfterSalesPayload,
    ContextualizedRequest,
    ConversationDST,
    CustomerServiceDomain,
    CustomerServiceIntent,
    DialogStatus,
    HandoffPayload,
    KnowledgePayload,
    OrderPayload,
    ProductPayload,
    SlotValue,
    TaskFrame,
)
from pydantic import BaseModel, ConfigDict, Field


class FSMAction(StrEnum):
    ASK_CLARIFICATION = "ask_clarification"
    COLLECT_SLOTS = "collect_slots"
    EXECUTE = "execute"
    WAIT_CONFIRMATION = "wait_confirmation"
    ANSWER = "answer"
    FAIL_CLOSED = "fail_closed"


class IntentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_slots: tuple[str, ...] = ()


class FSMDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: FSMAction
    business_action: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    message: str | None = None


_INTENT_SPECS = {
    CustomerServiceIntent.PRODUCT_RECOMMENDATION: IntentSpec(),
    CustomerServiceIntent.PRODUCT_SEARCH: IntentSpec(),
    CustomerServiceIntent.PRODUCT_REALTIME_FACT: IntentSpec(
        required_slots=("product_ref", "attributes")
    ),
    CustomerServiceIntent.PRODUCT_DOCUMENT_FACT: IntentSpec(
        required_slots=("product_ref", "attributes")
    ),
    CustomerServiceIntent.PRODUCT_COMPARISON: IntentSpec(required_slots=("product_refs",)),
    CustomerServiceIntent.ORDER_QUERY: IntentSpec(),
    CustomerServiceIntent.LOGISTICS_QUERY: IntentSpec(required_slots=("order_ref",)),
    CustomerServiceIntent.AFTER_SALES: IntentSpec(
        required_slots=(
            "order_ref",
            "customer_phone_last4",
            "issue_type",
            "reason",
        ),
    ),
    CustomerServiceIntent.HUMAN_HANDOFF: IntentSpec(
        required_slots=("order_ref", "customer_phone_last4", "reason")
    ),
    CustomerServiceIntent.POLICY_QUESTION: IntentSpec(),
    CustomerServiceIntent.GREETING: IntentSpec(),
    CustomerServiceIntent.OUT_OF_SCOPE: IntentSpec(),
    CustomerServiceIntent.OTHER: IntentSpec(),
}


def apply_request(dst: ConversationDST, request: ContextualizedRequest) -> None:
    if (
        dst.active_domain != CustomerServiceDomain.GENERAL
        and request.domain != dst.active_domain
        and dst.status
        not in {
            DialogStatus.IDLE,
            DialogStatus.COMPLETED,
            DialogStatus.FAILED,
        }
    ):
        dst.stack.append(
            TaskFrame(
                domain=dst.active_domain,
                intent=dst.active_intent,
                status=DialogStatus.INTERRUPTED,
                slots=dict(dst.slots),
            )
        )
        dst.stack = dst.stack[-10:]
    dst.active_domain = request.domain
    dst.active_intent = request.intent
    dst.raw_query = request.raw_query
    dst.rewritten_query = request.rewritten_query
    dst.error = None
    dst.slots = _slots_from_request(request)
    spec = _INTENT_SPECS[request.intent]
    dst.required_slots = list(spec.required_slots)
    dst.missing_slots = [
        name
        for name in dst.required_slots
        if name not in dst.slots or not dst.slots[name].validated
    ]
    if request.clarification_required:
        dst.status = DialogStatus.RESOLVING_TARGET
    elif (
        request.intent == CustomerServiceIntent.AFTER_SALES and dst.pending_confirmation is not None
    ):
        dst.status = DialogStatus.WAITING_CONFIRMATION
    elif dst.missing_slots:
        dst.status = DialogStatus.COLLECTING_SLOTS
    else:
        dst.status = DialogStatus.READY_TO_EXECUTE


def next_directive(
    dst: ConversationDST,
    request: ContextualizedRequest,
) -> FSMDirective:
    if request.clarification_required:
        return FSMDirective(
            action=FSMAction.ASK_CLARIFICATION,
            message=request.clarification_question,
        )
    if dst.status == DialogStatus.WAITING_CONFIRMATION:
        return FSMDirective(
            action=FSMAction.WAIT_CONFIRMATION,
            business_action=request.action,
        )
    if dst.missing_slots:
        return FSMDirective(
            action=FSMAction.COLLECT_SLOTS,
            missing_slots=list(dst.missing_slots),
        )
    return FSMDirective(
        action=FSMAction.EXECUTE,
        business_action=request.action,
    )


def _slots_from_request(
    request: ContextualizedRequest,
) -> dict[str, SlotValue]:
    slots: dict[str, SlotValue] = {}
    payload = request.payload
    if isinstance(payload, ProductPayload):
        if len(payload.target_product_codes) == 1:
            slots["product_ref"] = SlotValue(
                value=payload.target_product_codes[0],
                source=payload.resolution_source,
                validated=True,
            )
        elif payload.target_product_codes:
            slots["product_refs"] = SlotValue(
                value=list(payload.target_product_codes),
                source=payload.resolution_source,
                validated=True,
            )
        if payload.attributes:
            slots["attributes"] = SlotValue(
                value=list(payload.attributes),
                source=request.recognition_source,
                confidence=request.confidence,
                validated=True,
            )
        for key, value in payload.constraints.model_dump(exclude_none=True).items():
            if value in ([], ""):
                continue
            slots[key] = SlotValue(
                value=value,
                source=request.recognition_source,
                confidence=request.confidence,
                validated=True,
            )
    elif isinstance(payload, OrderPayload):
        if len(payload.target_order_refs) == 1:
            slots["order_ref"] = SlotValue(
                value=payload.target_order_refs[0],
                source=payload.resolution_source,
                validated=True,
            )
        elif payload.target_order_refs:
            slots["order_refs"] = SlotValue(
                value=list(payload.target_order_refs),
                source=payload.resolution_source,
                validated=True,
            )
    elif isinstance(payload, KnowledgePayload):
        slots["question"] = SlotValue(
            value=payload.question,
            source=request.recognition_source,
            confidence=request.confidence,
            validated=bool(payload.question.strip()),
        )
        if payload.product_ref:
            slots["product_ref"] = SlotValue(
                value=payload.product_ref,
                source=request.recognition_source,
                validated=True,
            )
        if payload.document_id is not None:
            slots["document_id"] = SlotValue(
                value=payload.document_id,
                source="trusted_binding",
                validated=True,
            )
    elif isinstance(payload, AfterSalesPayload):
        for name in (
            "order_ref",
            "customer_phone_last4",
            "issue_type",
            "reason",
        ):
            value = getattr(payload, name)
            if value:
                slots[name] = SlotValue(
                    value=value,
                    source=request.recognition_source,
                    confidence=request.confidence,
                    validated=True,
                )
    elif isinstance(payload, HandoffPayload):
        if payload.order_ref:
            slots["order_ref"] = SlotValue(
                value=payload.order_ref,
                source=request.recognition_source,
                validated=True,
            )
        if payload.customer_phone_last4:
            slots["customer_phone_last4"] = SlotValue(
                value=payload.customer_phone_last4,
                source=request.recognition_source,
                validated=True,
            )
        if payload.reason:
            slots["reason"] = SlotValue(
                value=payload.reason,
                source=request.recognition_source,
                confidence=request.confidence,
                validated=True,
            )
    return slots
