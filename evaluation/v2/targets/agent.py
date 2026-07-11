import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, cast

from backend.app.agents.factory import AgentRuntimeFactory
from backend.app.agents.state import AgentRuntimeRequest

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult
from evaluation.v2.targets.base import BaseEvaluationTarget, elapsed_ms


class AgentEvaluationTarget(BaseEvaluationTarget):
    name = "agent"

    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        started_at = perf_counter()
        runtime = AgentRuntimeFactory.get_runtime()
        request = AgentRuntimeRequest(
            query=case.query or str(case.input.get("query", "")),
            knowledge_base_id=case.input.get("knowledge_base_id"),
            conversation_id=case.input.get("conversation_id"),
            memory_context=case.input.get("memory_context"),
            metadata=case.input.get("metadata", {}),
        )
        try:
            arun = getattr(runtime, "arun", None)
            if callable(arun):
                async_run = cast(Callable[[AgentRuntimeRequest], Awaitable[Any]], arun)
                result = await async_run(request)
            else:
                result = await asyncio.to_thread(runtime.run, request)
            trace = [_dump(item) for item in result.trace]
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                answer=result.answer,
                output={"action": result.action},
                sources=result.sources,
                citations=result.citations,
                tool_calls=result.tool_calls,
                observations=result.observations,
                trace=trace,
                metadata=result.metadata,
                duration_ms=elapsed_ms(started_at),
            )
        except Exception as exc:
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                error=str(exc),
                duration_ms=elapsed_ms(started_at),
            )


def _dump(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"value": item}
