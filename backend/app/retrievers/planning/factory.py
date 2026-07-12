from functools import lru_cache

from backend.app.retrievers.planning.analyzer import FastQueryAnalyzer
from backend.app.retrievers.planning.constraint_engine import ConstraintEngine
from backend.app.retrievers.planning.constraint_registry import (
    ConstraintRegistry,
    default_constraint_registry,
)
from backend.app.retrievers.planning.planner import RetrievalPlanner


class RetrievalPlanningFactory:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_registry() -> ConstraintRegistry:
        return default_constraint_registry()

    @classmethod
    def get_analyzer(cls) -> FastQueryAnalyzer:
        return FastQueryAnalyzer(cls.get_registry())

    @classmethod
    def get_planner(cls) -> RetrievalPlanner:
        return RetrievalPlanner(cls.get_registry())

    @classmethod
    def get_constraint_engine(cls) -> ConstraintEngine:
        return ConstraintEngine(cls.get_registry())
