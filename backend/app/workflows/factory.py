from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.workflows.langgraph.runtime import LangGraphWorkflowRuntime
from backend.app.workflows.v1 import WorkflowRuntimeV1


class WorkflowRuntimeFactory:
    _v1_runtime: WorkflowRuntimeV1 | None = None
    _v2_runtime: LangGraphWorkflowRuntime | None = None

    @classmethod
    def get_runtime(cls, provider: str | None = None):
        selected = (provider or settings.WORKFLOW_RUNTIME).lower()
        if selected == "v1":
            if cls._v1_runtime is None:
                cls._v1_runtime = WorkflowRuntimeV1()
            return cls._v1_runtime
        if selected == "langgraph" and settings.WORKFLOW_V2_ENABLED:
            if cls._v2_runtime is None:
                cls._v2_runtime = LangGraphWorkflowRuntime()
            return cls._v2_runtime

        logger.warning(f"Unknown workflow runtime provider, fallback to v1: {selected}")
        if cls._v1_runtime is None:
            cls._v1_runtime = WorkflowRuntimeV1()
        return cls._v1_runtime

    @classmethod
    def reset(cls) -> None:
        cls._v1_runtime = None
        cls._v2_runtime = None
