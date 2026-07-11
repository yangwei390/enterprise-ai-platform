from abc import ABC, abstractmethod

from evaluation.v2.schemas import EvaluationCase, EvaluationTargetResult


class BaseJudge(ABC):
    name: str

    @abstractmethod
    async def judge(
        self,
        case: EvaluationCase,
        result: EvaluationTargetResult,
    ) -> dict:
        raise NotImplementedError
