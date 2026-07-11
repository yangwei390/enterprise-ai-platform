from evaluation.v2.judges.base import BaseJudge
from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class RuleBasedJudge(BaseJudge):
    name = "rule_based"

    async def judge(
        self,
        case: EvaluationCase,
        result: EvaluationTargetResult,
    ) -> dict:
        keywords = [str(item) for item in case.expected.get("keywords", [])]
        answer = result.answer or ""
        matched = [keyword for keyword in keywords if keyword in answer]
        score = 1.0 if not keywords else len(matched) / len(keywords)
        return {
            "score": score,
            "matched_keywords": matched,
            "proxy": True,
        }
