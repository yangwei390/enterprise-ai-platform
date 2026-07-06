from backend.app.schemas import ApiResponse, success
from backend.app.workflows import WorkflowDefinition, WorkflowService
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class WorkflowRunRequest(BaseModel):
    definition: WorkflowDefinition
    initial_state: dict | None = Field(default_factory=dict)


class WorkflowPlanAndRunRequest(BaseModel):
    task: str


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
