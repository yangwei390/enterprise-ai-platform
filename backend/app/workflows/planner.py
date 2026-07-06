import re
from uuid import uuid4

from backend.app.workflows.base import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowNodeType,
)


class SimpleWorkflowPlanner:
    def plan(self, task: str) -> WorkflowDefinition:
        if self._looks_like_calculation(task):
            return self._build_calculator_workflow(task)
        return self._build_llm_workflow(task)

    def _looks_like_calculation(self, task: str) -> bool:
        return "计算" in task or bool(re.search(r"\d+\s*[+\-*/%]\s*\d+", task))

    def _build_calculator_workflow(self, task: str) -> WorkflowDefinition:
        expression = self._extract_expression(task)
        workflow_id = f"workflow_{uuid4().hex[:8]}"
        return WorkflowDefinition(
            id=workflow_id,
            name="calculator workflow",
            description="Auto planned calculator workflow.",
            start_node_id="start",
            nodes=[
                WorkflowNode(id="start", type=WorkflowNodeType.START, next=["calculator"]),
                WorkflowNode(
                    id="calculator",
                    type=WorkflowNodeType.TOOL,
                    config={
                        "tool_name": "calculator",
                        "arguments": {"expression": expression},
                        "output_key": "calc_result",
                    },
                    next=["end"],
                ),
                WorkflowNode(id="end", type=WorkflowNodeType.END),
            ],
        )

    def _build_llm_workflow(self, task: str) -> WorkflowDefinition:
        workflow_id = f"workflow_{uuid4().hex[:8]}"
        return WorkflowDefinition(
            id=workflow_id,
            name="llm workflow",
            description="Auto planned LLM workflow.",
            start_node_id="start",
            nodes=[
                WorkflowNode(id="start", type=WorkflowNodeType.START, next=["llm"]),
                WorkflowNode(
                    id="llm",
                    type=WorkflowNodeType.LLM,
                    config={
                        "prompt": task,
                        "output_key": "llm_result",
                    },
                    next=["end"],
                ),
                WorkflowNode(id="end", type=WorkflowNodeType.END),
            ],
        )

    def _extract_expression(self, task: str) -> str:
        allowed_parts = re.findall(r"[0-9+\-*/%().\s]+", task)
        expression = "".join(allowed_parts).strip()
        return expression or task.replace("计算", "").strip()
