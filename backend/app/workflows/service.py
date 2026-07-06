from backend.app.workflows.base import WorkflowDefinition, WorkflowResult
from backend.app.workflows.planner import SimpleWorkflowPlanner
from backend.app.workflows.runtime import WorkflowRuntime


class WorkflowService:
    def __init__(
        self,
        runtime: WorkflowRuntime | None = None,
        planner: SimpleWorkflowPlanner | None = None,
    ) -> None:
        self.runtime = runtime or WorkflowRuntime()
        self.planner = planner or SimpleWorkflowPlanner()

    def run_workflow(
        self,
        definition: WorkflowDefinition,
        initial_state: dict | None = None,
    ) -> WorkflowResult:
        return self.runtime.run(definition, initial_state=initial_state)

    def plan_and_run(self, task: str) -> WorkflowResult:
        definition = self.planner.plan(task)
        return self.run_workflow(definition)
