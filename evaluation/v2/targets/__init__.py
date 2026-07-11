from evaluation.v2.targets.agent import AgentEvaluationTarget
from evaluation.v2.targets.base import BaseEvaluationTarget
from evaluation.v2.targets.generation import GenerationEvaluationTarget
from evaluation.v2.targets.mcp import MCPEvaluationTarget
from evaluation.v2.targets.rag import RAGEvaluationTarget
from evaluation.v2.targets.tool import ToolEvaluationTarget
from evaluation.v2.targets.workflow import WorkflowEvaluationTarget


def get_target(name: str) -> BaseEvaluationTarget:
    targets: dict[str, BaseEvaluationTarget] = {
        "rag": RAGEvaluationTarget(),
        "generation": GenerationEvaluationTarget(),
        "agent": AgentEvaluationTarget(),
        "tool": ToolEvaluationTarget(),
        "mcp": MCPEvaluationTarget(),
        "workflow": WorkflowEvaluationTarget(),
    }
    return targets[name]


def list_targets() -> list[str]:
    return ["rag", "generation", "agent", "tool", "mcp", "workflow"]


__all__ = ["get_target", "list_targets", "BaseEvaluationTarget"]
