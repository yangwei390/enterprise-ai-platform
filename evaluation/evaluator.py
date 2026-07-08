from dataclasses import dataclass, field
from typing import Any

from backend.app.chat import ChatResponse
from backend.app.config.settings import settings

from evaluation.metrics import chunk_recall, document_hit, keyword_coverage


@dataclass
class EvaluationQuestion:
    id: str
    question: str
    expected_documents: list[str] = field(default_factory=list)
    expected_chunks: list[int] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    knowledge_base_id: int | None = None
    top_k: int = 5
    score_threshold: float | None = None


class Evaluator:
    def __init__(self, keyword_threshold: float | None = None) -> None:
        self.keyword_threshold = (
            settings.EVALUATION_KEYWORD_THRESHOLD
            if keyword_threshold is None
            else keyword_threshold
        )

    def evaluate(
        self,
        question: EvaluationQuestion,
        response: ChatResponse,
        latency: dict[str, float],
    ) -> dict[str, Any]:
        retriever_hit = document_hit(question.expected_documents, response.sources)
        recall = chunk_recall(question.expected_chunks, response.sources)
        coverage = keyword_coverage(question.expected_keywords, response.answer)
        passed = retriever_hit and coverage >= self.keyword_threshold

        return {
            "id": question.id,
            "question": question.question,
            "pass": passed,
            "retriever_hit": retriever_hit,
            "chunk_recall": recall,
            "keyword_coverage": coverage,
            "latency_ms": latency.get("total_ms", 0.0),
            "retriever_latency_ms": latency.get("retriever_ms", 0.0),
            "llm_latency_ms": latency.get("llm_ms", 0.0),
            "expected_documents": question.expected_documents,
            "expected_chunks": question.expected_chunks,
            "expected_keywords": question.expected_keywords,
            "sources": [
                source.model_dump() if hasattr(source, "model_dump") else source
                for source in response.sources
            ],
            "metadata": response.metadata,
        }
