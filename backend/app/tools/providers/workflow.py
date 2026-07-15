import asyncio

from backend.app.config.settings import settings
from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.providers.base import BaseToolProvider
from backend.app.workflows.factory import WorkflowRuntimeFactory
from backend.app.workflows.langgraph import WorkflowRunRequestV2, WorkflowRunResultV2
from pydantic import BaseModel, Field


class WorkflowToolArgs(BaseModel):
    query: str
    knowledge_base_id: int | None = None
    inputs: dict = Field(default_factory=dict)


class WorkflowTool(BaseTool):
    name = "workflow_default_knowledge"
    description = "运行默认知识库问答 Workflow。"
    args_schema = WorkflowToolArgs
    source = "workflow"
    permission = "public"

    def __init__(self, runtime=None) -> None:
        self.runtime = runtime or WorkflowRuntimeFactory.get_runtime()

    def run(self, arguments: dict) -> ToolResult:
        args = WorkflowToolArgs.model_validate(arguments)
        result = self.runtime.run(
            WorkflowRunRequestV2(
                workflow_id="default_agent_workflow_v2",
                query=args.query,
                knowledge_base_id=args.knowledge_base_id,
                inputs=args.inputs,
            )
        )
        return ToolResult(
            name=self.name,
            success=_workflow_succeeded(result),
            result=result.model_dump(),
            error=_workflow_error(result),
            metadata={
                "provider": "workflow",
                "workflow_id": "default_agent_workflow_v2",
            },
        )

    async def arun(self, arguments: dict) -> ToolResult:
        return await asyncio.to_thread(self.run, arguments)


def _workflow_succeeded(result: WorkflowRunResultV2) -> bool:
    return result.status in {"completed", "success"}


def _workflow_error(result: WorkflowRunResultV2) -> str | None:
    workflow_runtime = result.metadata.get("workflow_runtime", {})
    return workflow_runtime.get("error") or result.metadata.get("error")


class WorkflowToolProvider(BaseToolProvider):
    @property
    def name(self) -> str:
        return "workflow"

    def discover(self) -> list[BaseTool]:
        if not settings.WORKFLOW_TOOL_PROVIDER_ENABLED:
            return []
        return [WorkflowTool()]
