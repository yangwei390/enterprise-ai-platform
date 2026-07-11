from backend.app.tools import get_tool_registry

from evaluation.v2.schemas import EvaluationCase, EvaluationContext, EvaluationTargetResult
from evaluation.v2.targets.tool import ToolEvaluationTarget


class MCPEvaluationTarget(ToolEvaluationTarget):
    name = "mcp"

    async def arun(
        self,
        case: EvaluationCase,
        context: EvaluationContext,
    ) -> EvaluationTargetResult:
        tool_name = str(case.input.get("tool_name") or "")
        registry = get_tool_registry()
        descriptor = registry.get_descriptor(tool_name)
        if descriptor is None or descriptor.provider != "mcp":
            return EvaluationTargetResult(
                target=self.name,
                input=case.input,
                skipped=True,
                skip_reason="mcp tool unavailable",
                metadata={"mcp_tool_available": False},
            )
        result = await super().arun(case, context)
        result.target = self.name
        result.metadata["mcp_tool_available"] = True
        return result
