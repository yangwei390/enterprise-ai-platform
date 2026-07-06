from backend.app.agents.base import (
    AgentDefinition,
    AgentRunRequest,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    AgentStepType,
)
from backend.app.agents.planner import AgentPlanner
from backend.app.agents.reflection import AgentReflection
from backend.app.workflows import WorkflowResult, WorkflowService


class AgentExecutor:
    def __init__(
        self,
        definition: AgentDefinition,
        planner: AgentPlanner | None = None,
        workflow_service: WorkflowService | None = None,
        reflection: AgentReflection | None = None,
    ) -> None:
        self.definition = definition
        self.planner = planner or AgentPlanner()
        self.workflow_service = workflow_service or WorkflowService()
        self.reflection = reflection or AgentReflection()

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        steps: list[AgentStep] = []
        artifacts = []
        metadata: dict = {"agent": self.definition.model_dump()}
        error: str | None = None

        try:
            plan_step = self._new_step(
                steps,
                AgentStepType.PLAN,
                "plan workflow",
                {"task": request.task},
            )
            workflow_definition = self.planner.plan(request)
            plan_step.output = {"workflow_definition": workflow_definition.model_dump()}
            plan_step.status = AgentRunStatus.SUCCESS

            workflow_step = self._new_step(
                steps,
                AgentStepType.WORKFLOW,
                "run workflow",
                {"workflow_id": workflow_definition.id},
            )
            workflow_result = self.workflow_service.run_workflow(
                workflow_definition,
                initial_state={"query": request.task},
            )
            workflow_step.output = {"workflow_result": workflow_result.model_dump()}
            workflow_step.status = (
                AgentRunStatus.SUCCESS
                if workflow_result.status == "success"
                else AgentRunStatus.FAILED
            )
            workflow_step.error = workflow_result.error
            artifacts = workflow_result.artifacts
            metadata["workflow"] = {
                "workflow_id": workflow_result.workflow_id,
                "status": workflow_result.status,
                "logs": workflow_result.logs,
            }

            if self.definition.enable_reflection:
                reflection_step = self._new_step(
                    steps,
                    AgentStepType.REFLECTION,
                    "reflect workflow result",
                    {"workflow_id": workflow_result.workflow_id},
                )
                reflection_result = self.reflection.reflect(request.task, workflow_result)
                reflection_step.output = reflection_result
                reflection_step.status = AgentRunStatus.SUCCESS
                metadata["reflection"] = reflection_result

            answer = self._build_final_answer(workflow_result)
            final_step = self._new_step(
                steps,
                AgentStepType.FINAL,
                "final answer",
                {"workflow_id": workflow_result.workflow_id},
            )
            final_step.output = {"answer": answer}
            final_step.status = AgentRunStatus.SUCCESS

            status = (
                AgentRunStatus.SUCCESS
                if workflow_result.status == "success"
                else AgentRunStatus.FAILED
            )
            error = workflow_result.error
        except Exception as exc:
            status = AgentRunStatus.FAILED
            answer = None
            error = str(exc)
            if steps and steps[-1].status == AgentRunStatus.PENDING:
                steps[-1].status = AgentRunStatus.FAILED
                steps[-1].error = error

        return AgentRunResult(
            task=request.task,
            status=status,
            answer=answer,
            steps=steps,
            artifacts=artifacts,
            metadata=metadata,
            error=error,
        )

    def _new_step(
        self,
        steps: list[AgentStep],
        step_type: AgentStepType,
        name: str,
        input_data: dict,
    ) -> AgentStep:
        step = AgentStep(
            index=len(steps),
            type=step_type,
            name=name,
            input=input_data,
            status=AgentRunStatus.RUNNING,
        )
        steps.append(step)
        return step

    def _build_final_answer(self, workflow_result: WorkflowResult) -> str:
        llm_result = workflow_result.state.values.get("llm_result")
        tool_summary = self._build_tool_result_summary(workflow_result)
        if isinstance(llm_result, dict):
            answer = llm_result.get("answer")
            if answer:
                return self._append_tool_summary(str(answer), tool_summary)

        for artifact in workflow_result.artifacts:
            if artifact.key == "llm_result" and isinstance(artifact.value, dict):
                answer = artifact.value.get("answer")
                if answer:
                    return self._append_tool_summary(str(answer), tool_summary)

        if tool_summary:
            return tool_summary

        return "任务已执行完成，但没有生成最终回答。"

    def _build_tool_result_summary(self, workflow_result: WorkflowResult) -> str | None:
        summaries = []
        for artifact in workflow_result.artifacts:
            if artifact.key == "llm_result":
                continue
            if artifact.key.endswith("_result"):
                summaries.append(f"{artifact.key}: {artifact.value}")
        return "\n".join(summaries) if summaries else None

    def _append_tool_summary(self, answer: str, tool_summary: str | None) -> str:
        if not tool_summary:
            return answer
        return f"{answer}\n\n工具结果：\n{tool_summary}"
