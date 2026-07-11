import hashlib
import json
from time import perf_counter

from backend.app.tools import ToolCall, ToolExecutor

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult
from evaluation.v2.targets.base import BaseEvaluationTarget, elapsed_ms


class ToolEvaluationTarget(BaseEvaluationTarget):
    name = "tool"

    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        started_at = perf_counter()
        tool_name = str(case.input.get("tool_name") or "")
        arguments = case.input.get("arguments", {})
        result = await ToolExecutor().aexecute(ToolCall(name=tool_name, arguments=arguments))
        return EvaluationTargetResult(
            target=self.name,
            input={"tool_name": tool_name, "arguments_hash": _arguments_hash(arguments)},
            output={"success": result.success, "result": result.result, "error": result.error},
            answer=str(result.result) if result.result is not None else None,
            metadata=result.metadata,
            duration_ms=elapsed_ms(started_at),
            error=result.error if not result.success else None,
        )


def _arguments_hash(arguments: object) -> str:
    raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
