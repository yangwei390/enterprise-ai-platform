from dataclasses import dataclass

from backend.app.config.settings import settings
from backend.app.documents.schemas import DocumentStructure


@dataclass
class ChunkStrategyDecision:
    requested_strategy: str
    actual_strategy: str
    reason: str

    def to_metadata(self) -> dict:
        return {
            "chunk_strategy_requested": self.requested_strategy,
            "chunk_strategy_actual": self.actual_strategy,
            "chunk_strategy_router_reason": self.reason,
        }


class ChunkStrategyRouter:
    valid_strategies = {
        "fixed",
        "auto",
        "recursive",
        "markdown",
        "legal",
        "legal_structure",
        "semantic",
        "parent_child",
    }

    def route(
        self,
        *,
        document_type: str,
        structure: DocumentStructure | None = None,
        metadata: dict | None = None,
        requested_strategy: str | None = None,
    ) -> ChunkStrategyDecision:
        requested = (requested_strategy or settings.CHUNK_STRATEGY or "auto").lower()
        if requested != "auto" and requested in self.valid_strategies:
            actual = "legal_structure" if requested == "legal" else requested
            return ChunkStrategyDecision(requested, actual, "explicit_strategy")
        if document_type == "legal":
            return ChunkStrategyDecision(requested, "legal_structure", "document_type=legal")
        if document_type == "markdown":
            return ChunkStrategyDecision(requested, "markdown", "document_type=markdown")
        if settings.CHUNK_SEMANTIC_ENABLED and document_type == "plain_text":
            return ChunkStrategyDecision(requested, "semantic", "semantic_enabled")
        return ChunkStrategyDecision(requested, "recursive", f"document_type={document_type}")
