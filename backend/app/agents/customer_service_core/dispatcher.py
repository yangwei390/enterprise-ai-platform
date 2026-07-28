from __future__ import annotations

from enum import StrEnum

from backend.app.agents.customer_service_core.actions import allowed_tools
from backend.app.agents.customer_service_core.fsm import FSMAction, FSMDirective
from backend.app.agents.customer_service_core.schemas import (
    ContextualizedRequest,
    CustomerServiceDomain,
    CustomerServiceIntent,
)
from pydantic import BaseModel, ConfigDict, Field


class DispatchPhase(StrEnum):
    CLARIFY = "clarify"
    COLLECT = "collect"
    WAIT = "wait"
    EXECUTE = "execute"


class DispatchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: DispatchPhase
    domain: CustomerServiceDomain
    intent: CustomerServiceIntent
    business_action: str
    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    missing_slots: tuple[str, ...] = ()
    message: str | None = None


def build_dispatch_plan(
    request: ContextualizedRequest,
    directive: FSMDirective,
) -> DispatchPlan:
    phase = {
        FSMAction.ASK_CLARIFICATION: DispatchPhase.CLARIFY,
        FSMAction.COLLECT_SLOTS: DispatchPhase.COLLECT,
        FSMAction.WAIT_CONFIRMATION: DispatchPhase.WAIT,
        FSMAction.EXECUTE: DispatchPhase.EXECUTE,
        FSMAction.ANSWER: DispatchPhase.EXECUTE,
        FSMAction.FAIL_CLOSED: DispatchPhase.CLARIFY,
    }[directive.action]
    return DispatchPlan(
        phase=phase,
        domain=request.domain,
        intent=request.intent,
        business_action=directive.business_action or request.action,
        allowed_tools=allowed_tools(request.intent),
        missing_slots=tuple(directive.missing_slots),
        message=directive.message,
    )
