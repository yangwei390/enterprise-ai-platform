from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from backend.app.agents.customer_service_core.schemas import ConversationDST


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class CompatibilityResult:
    def __init__(
        self,
        *,
        status: CompatibilityStatus,
        slot: str,
        reason: str,
        source: str,
    ) -> None:
        self.status = status
        self.slot = slot
        self.reason = reason
        self.source = source


class CompatibilityProvider(Protocol):
    def evaluate(
        self,
        *,
        dst: ConversationDST,
        category: str,
        slot: str,
        value: object,
    ) -> CompatibilityResult: ...


class ConservativeCompatibilityProvider:
    _ALWAYS_COMPATIBLE = {"brand", "price_min", "price_max"}

    def evaluate(
        self,
        *,
        dst: ConversationDST,
        category: str,
        slot: str,
        value: object,
    ) -> CompatibilityResult:
        if slot in self._ALWAYS_COMPATIBLE:
            return CompatibilityResult(
                status=CompatibilityStatus.COMPATIBLE,
                slot=slot,
                reason="slot_is_category_independent",
                source="compatibility_rules",
            )
        return CompatibilityResult(
            status=CompatibilityStatus.UNKNOWN,
            slot=slot,
            reason="category_compatibility_unknown",
            source="compatibility_rules",
        )


DEFAULT_COMPATIBILITY_PROVIDER: CompatibilityProvider = ConservativeCompatibilityProvider()
