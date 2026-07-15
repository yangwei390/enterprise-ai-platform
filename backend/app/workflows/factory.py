from backend.app.logger import logger
from backend.app.workflows.langgraph.runtime import LangGraphWorkflowRuntime


class WorkflowRuntimeFactory:
    _runtime: LangGraphWorkflowRuntime | None = None

    @classmethod
    def get_runtime(cls, provider: str | None = None):
        selected = (provider or "langgraph").lower()
        if selected not in {"langgraph", "v2", "langgraph_v2"}:
            logger.warning("Unknown workflow runtime provider, using LangGraph V2: {}", selected)
        if cls._runtime is None:
            cls._runtime = LangGraphWorkflowRuntime()
        return cls._runtime

    @classmethod
    def reset(cls) -> None:
        cls._runtime = None
