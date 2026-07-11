import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, cast

from backend.app.workflows.factory import WorkflowRuntimeFactory
from backend.app.workflows.langgraph import WorkflowRunRequestV2
from backend.app.workflows.v1 import WorkflowRunRequest

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult
from evaluation.v2.targets.base import BaseEvaluationTarget, elapsed_ms


class WorkflowEvaluationTarget(BaseEvaluationTarget):
    name = "workflow"

    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        started_at = perf_counter()
        runtime = WorkflowRuntimeFactory.get_runtime()
        workflow_id = case.input.get("workflow_id")
        try:
            arun = getattr(runtime, "arun", None)
            if callable(arun):
                async_run = cast(Callable[[WorkflowRunRequestV2], Awaitable[Any]], arun)
                result = await async_run(
                    WorkflowRunRequestV2(
                        workflow_id=workflow_id,
                        query=case.query or str(case.input.get("query", "")),
                        knowledge_base_id=case.input.get("knowledge_base_id"),
                        inputs=case.input.get("inputs", {}),
                        metadata=case.input.get("metadata", {}),
                    )
                )
            else:
                result = await asyncio.to_thread(
                    runtime.run,
                    WorkflowRunRequest(
                        workflow_id=workflow_id,
                        query=case.query or str(case.input.get("query", "")),
                        knowledge_base_id=case.input.get("knowledge_base_id"),
                        inputs=case.input.get("inputs", {}),
                    ),
                )
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                answer=getattr(result, "answer", None),
                output=getattr(result, "output", {}),
                trace=[_dump(item) for item in getattr(result, "trace", [])],
                metadata=getattr(result, "metadata", {}),
                duration_ms=elapsed_ms(started_at),
                error=None if _status_ok(result) else getattr(result, "error", None),
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


def _status_ok(result: Any) -> bool:
    status = getattr(result, "status", None)
    metadata = getattr(result, "metadata", {})
    return status in (None, "completed", "success") and not metadata.get("failed")
