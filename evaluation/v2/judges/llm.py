import asyncio
from time import perf_counter

from backend.app.config.settings import settings
from backend.app.llms import LLMFactory, LLMRequest

from evaluation.v2.errors import EvaluationError
from evaluation.v2.judges.base import BaseJudge
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class LLMJudge(BaseJudge):
    name = "llm"

    async def judge(
        self,
        case: EvaluationCase,
        result: EvaluationTargetResult,
    ) -> dict:
        if not settings.EVALUATION_LLM_JUDGE_ENABLED:
            return {"enabled": False, "score": None}

        prompt = (
            "你是企业级评测器。只根据问题、期望和回答给出 0 到 1 的完成度分数。"
            '输出 JSON: {"score": number, "reason": string}。\n\n'
            f"问题: {case.query or case.input.get('prompt', '')}\n"
            f"期望: {case.expected}\n"
            f"回答: {(result.answer or '')[: settings.EVALUATION_REPORT_MAX_TEXT_CHARS]}"
        )
        started_at = perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    LLMFactory.get_llm().generate,
                    LLMRequest(
                        prompt=prompt,
                        model=settings.EVALUATION_LLM_JUDGE_MODEL,
                        temperature=0,
                    ),
                ),
                timeout=settings.EVALUATION_LLM_JUDGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise EvaluationError(f"LLM judge failed: {exc}") from exc

        return {
            "enabled": True,
            "score": None,
            "raw": response.content,
            "judge_model": settings.EVALUATION_LLM_JUDGE_MODEL,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        }
