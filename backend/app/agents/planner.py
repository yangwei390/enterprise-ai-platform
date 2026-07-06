import re
from uuid import uuid4

from backend.app.agents.base import AgentRunRequest
from backend.app.workflows import WorkflowDefinition, WorkflowNode, WorkflowNodeType


class AgentPlanner:
    def plan(self, request: AgentRunRequest) -> WorkflowDefinition:
        task = request.task
        if self._looks_like_calculation(task):
            return self._build_calculator_workflow(task)
        if self._looks_like_knowledge_task(task):
            return self._build_retriever_workflow(request)
        if "工具" in task:
            return self._build_echo_tool_workflow(task)
        return self._build_llm_workflow(task)

    def _looks_like_calculation(self, task: str) -> bool:
        return "计算" in task or bool(re.search(r"\d+\s*[+\-*/%]\s*\d+", task))

    def _looks_like_knowledge_task(self, task: str) -> bool:
        keywords = ["知识库", "文档", "是什么", "解释"]
        return any(keyword in task for keyword in keywords)

    def _build_calculator_workflow(self, task: str) -> WorkflowDefinition:
        expression = self._extract_expression(task)
        return WorkflowDefinition(
            id=self._workflow_id(),
            name="agent calculator workflow",
            description="Agent planned calculator workflow.",
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
                    next=["llm_summary"],
                ),
                WorkflowNode(
                    id="llm_summary",
                    type=WorkflowNodeType.LLM,
                    config={
                        "prompt": "请用自然语言总结这个计算结果：{{calc_result}}",
                        "output_key": "llm_result",
                    },
                    next=["end"],
                ),
                WorkflowNode(id="end", type=WorkflowNodeType.END),
            ],
        )

    def _build_retriever_workflow(self, request: AgentRunRequest) -> WorkflowDefinition:
        return WorkflowDefinition(
            id=self._workflow_id(),
            name="agent retriever workflow",
            description="Agent planned retriever workflow.",
            start_node_id="start",
            nodes=[
                WorkflowNode(id="start", type=WorkflowNodeType.START, next=["retriever"]),
                WorkflowNode(
                    id="retriever",
                    type=WorkflowNodeType.RETRIEVER,
                    config={
                        "query": request.task,
                        "knowledge_base_id": request.knowledge_base_id,
                        "top_k": 5,
                        "output_key": "retrieval_result",
                    },
                    next=["llm_summary"],
                ),
                WorkflowNode(
                    id="llm_summary",
                    type=WorkflowNodeType.LLM,
                    config={
                        "prompt": (
                            "请基于以下检索结果回答用户问题。\n"
                            "问题："
                            f"{request.task}\n"
                            "检索结果：{{retrieval_result}}"
                        ),
                        "output_key": "llm_result",
                    },
                    next=["end"],
                ),
                WorkflowNode(id="end", type=WorkflowNodeType.END),
            ],
        )

    def _build_echo_tool_workflow(self, task: str) -> WorkflowDefinition:
        return WorkflowDefinition(
            id=self._workflow_id(),
            name="agent echo tool workflow",
            description="Agent planned echo tool workflow.",
            start_node_id="start",
            nodes=[
                WorkflowNode(id="start", type=WorkflowNodeType.START, next=["echo"]),
                WorkflowNode(
                    id="echo",
                    type=WorkflowNodeType.TOOL,
                    config={
                        "tool_name": "echo",
                        "arguments": {"text": task},
                        "output_key": "echo_result",
                    },
                    next=["llm_summary"],
                ),
                WorkflowNode(
                    id="llm_summary",
                    type=WorkflowNodeType.LLM,
                    config={
                        "prompt": "请总结工具执行结果：{{echo_result}}",
                        "output_key": "llm_result",
                    },
                    next=["end"],
                ),
                WorkflowNode(id="end", type=WorkflowNodeType.END),
            ],
        )

    def _build_llm_workflow(self, task: str) -> WorkflowDefinition:
        return WorkflowDefinition(
            id=self._workflow_id(),
            name="agent llm workflow",
            description="Agent planned LLM workflow.",
            start_node_id="start",
            nodes=[
                WorkflowNode(id="start", type=WorkflowNodeType.START, next=["llm"]),
                WorkflowNode(
                    id="llm",
                    type=WorkflowNodeType.LLM,
                    config={"prompt": task, "output_key": "llm_result"},
                    next=["end"],
                ),
                WorkflowNode(id="end", type=WorkflowNodeType.END),
            ],
        )

    def _extract_expression(self, task: str) -> str:
        allowed_parts = re.findall(r"[0-9+\-*/%().\s]+", task)
        expression = "".join(allowed_parts).strip()
        return expression or task.replace("计算", "").strip()

    def _workflow_id(self) -> str:
        return f"agent_workflow_{uuid4().hex[:8]}"
