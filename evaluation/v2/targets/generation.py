import asyncio
from time import perf_counter

from backend.app.llms import LLMFactory, LLMMessage, LLMRequest

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult
from evaluation.v2.targets.base import BaseEvaluationTarget, elapsed_ms


class GenerationEvaluationTarget(BaseEvaluationTarget):
    name = "generation"

    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        started_at = perf_counter()
        messages = []
        if case.input.get("system_prompt"):
            messages.append(LLMMessage(role="system", content=str(case.input["system_prompt"])))
        prompt = str(case.input.get("prompt") or case.query or "")
        messages.append(LLMMessage(role="user", content=prompt))
        try:
            response = await asyncio.to_thread(
                LLMFactory.get_llm().chat,
                LLMRequest(messages=messages, model=case.input.get("model")),
            )
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                answer=response.answer,
                output={"model": response.model},
                metadata=response.metadata,
                usage=response.usage,
                duration_ms=elapsed_ms(started_at),
            )
        except Exception as exc:
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                error=str(exc),
                duration_ms=elapsed_ms(started_at),
            )
