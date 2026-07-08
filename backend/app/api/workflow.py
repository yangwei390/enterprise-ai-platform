from backend.app.schemas import ApiResponse, success
from backend.app.workflows import (
    WorkflowDefinition,
    WorkflowRuntimeV1,
    WorkflowService,
)
from backend.app.workflows import (
    WorkflowRunRequest as WorkflowRunV1Request,
)
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class WorkflowRunRequest(BaseModel):
    definition: WorkflowDefinition
    initial_state: dict | None = Field(default_factory=dict)


class WorkflowPlanAndRunRequest(BaseModel):
    task: str


@router.post("/workflow/run", response_model=ApiResponse)
def run_workflow_v1(request: WorkflowRunV1Request) -> ApiResponse:
    result = WorkflowRuntimeV1().run(request)
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
