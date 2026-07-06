from backend.app.workflows import WorkflowResult


class AgentReflection:
    def reflect(self, task: str, workflow_result: WorkflowResult) -> dict:
        if workflow_result.status != "success":
            return {
                "passed": False,
                "reason": "workflow failed",
                "suggestion": "check workflow logs and failed node",
            }

        if not workflow_result.artifacts:
            return {
                "passed": False,
                "reason": "no artifacts",
                "suggestion": "ensure workflow nodes write artifacts",
            }

        if workflow_result.state.values.get("llm_result") is not None:
            return {
                "passed": True,
                "reason": "llm_result generated",
                "suggestion": "",
            }

        for artifact in workflow_result.artifacts:
            if artifact.key == "llm_result":
                return {
                    "passed": True,
                    "reason": "llm_result artifact generated",
                    "suggestion": "",
                }

        return {
            "passed": False,
            "reason": "no final answer artifact",
            "suggestion": "add an LLM summary node or final answer node",
        }
