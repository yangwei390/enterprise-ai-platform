import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from backend.app.config.settings import settings
from backend.app.schemas import ApiResponse, success
from backend.app.workflows import (
    WorkflowDefinition,
    WorkflowService,
)
from backend.app.workflows import (
    WorkflowRunRequest as WorkflowRunV1Request,
)
from backend.app.workflows.factory import WorkflowRuntimeFactory
from backend.app.workflows.langgraph import WorkflowResumeRequest, WorkflowRunRequestV2
from backend.app.workflows.langgraph.runtime import LangGraphWorkflowRuntime
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class WorkflowRunRequest(BaseModel):
    definition: WorkflowDefinition
    initial_state: dict | None = Field(default_factory=dict)


class WorkflowPlanAndRunRequest(BaseModel):
    task: str


@router.post("/workflow/run", response_model=ApiResponse)
async def run_workflow_v1(request: WorkflowRunV1Request) -> ApiResponse:
    runtime = WorkflowRuntimeFactory.get_runtime()
    arun = getattr(runtime, "arun", None)
    if callable(arun):
        async_run = cast(Callable[[WorkflowRunRequestV2], Awaitable[Any]], arun)
        workflow_id = request.workflow_id
        if settings.WORKFLOW_RUNTIME.lower() == "langgraph" and workflow_id in (
            None,
            "default_knowledge_workflow",
        ):
            workflow_id = "default_agent_workflow_v2"
        result = await async_run(
            WorkflowRunRequestV2(
                workflow_id=workflow_id,
                query=request.query,
                knowledge_base_id=request.knowledge_base_id,
                inputs=request.inputs,
                thread_id=request.thread_id,
                metadata=request.metadata,
            )
        )
    else:
        sync_run = cast(Callable[[WorkflowRunV1Request], Any], runtime.run)
        result = await asyncio.to_thread(sync_run, request)
    return success(data=result)


@router.post("/workflow/resume", response_model=ApiResponse)
async def resume_workflow(request: WorkflowResumeRequest) -> ApiResponse:
    runtime = cast(LangGraphWorkflowRuntime, WorkflowRuntimeFactory.get_runtime("langgraph"))
    result = await runtime.aresume(request)
    return success(data=result)


@router.post("/workflows/run", response_model=ApiResponse)
def run_workflow(request: WorkflowRunRequest) -> ApiResponse:
    result = WorkflowService().run_workflow(
        definition=request.definition,
        initial_state=request.initial_state,
    )
    return success(data=result)


@router.post("/workflows/plan-and-run", response_model=ApiResponse)
def plan_and_run_workflow(request: WorkflowPlanAndRunRequest) -> ApiResponse:
    result = WorkflowService().plan_and_run(request.task)
    return success(data=result)
