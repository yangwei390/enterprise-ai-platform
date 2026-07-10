import asyncio

from backend.app.config.settings import settings
from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.providers.base import BaseToolProvider
from backend.app.workflows.v1 import WorkflowRunRequest, WorkflowRuntimeV1
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

    def __init__(self, runtime: WorkflowRuntimeV1 | None = None) -> None:
        self.runtime = runtime or WorkflowRuntimeV1()

    def run(self, arguments: dict) -> ToolResult:
        args = WorkflowToolArgs.model_validate(arguments)
        result = self.runtime.run(
            WorkflowRunRequest(
                workflow_id="default_knowledge_workflow",
                query=args.query,
                knowledge_base_id=args.knowledge_base_id,
                inputs=args.inputs,
            )
        )
        return ToolResult(
            name=self.name,
            success=not bool(result.metadata.get("failed")),
            result=result.model_dump(),
            error=result.metadata.get("error"),
            metadata={
                "provider": "workflow",
                "workflow_id": "default_knowledge_workflow",
            },
        )

    async def arun(self, arguments: dict) -> ToolResult:
        return await asyncio.to_thread(self.run, arguments)


class WorkflowToolProvider(BaseToolProvider):
    @property
    def name(self) -> str:
        return "workflow"

    def discover(self) -> list[BaseTool]:
        if not settings.WORKFLOW_TOOL_PROVIDER_ENABLED:
            return []
        return [WorkflowTool()]
