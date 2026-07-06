import re
from abc import ABC, abstractmethod
from typing import Any

from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.retrievers import RetrieverFactory
from backend.app.retrievers.hybrid import HybridRetrieveQuery
from backend.app.tools import ToolCall, ToolExecutor
from backend.app.workflows.base import WorkflowNode, WorkflowNodeType, WorkflowStatus
from backend.app.workflows.context import WorkflowContext


class BaseWorkflowNodeExecutor(ABC):
    @abstractmethod
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        raise NotImplementedError


class StartNodeExecutor(BaseWorkflowNodeExecutor):
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        return node.next[0] if node.next else None


class EndNodeExecutor(BaseWorkflowNodeExecutor):
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        context.state.status = WorkflowStatus.SUCCESS
        return None


class ToolNodeExecutor(BaseWorkflowNodeExecutor):
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        tool_name = str(node.config.get("tool_name", ""))
        arguments = node.config.get("arguments", {})
        output_key = str(node.config.get("output_key", f"{node.id}_result"))
        result = ToolExecutor().execute(
            ToolCall(name=tool_name, arguments=arguments if isinstance(arguments, dict) else {})
        )
        result_data = result.model_dump()
        context.set_value(output_key, result_data)
        context.add_artifact(output_key, result_data, {"node_id": node.id, "type": node.type})
        return node.next[0] if node.next else None


class RetrieverNodeExecutor(BaseWorkflowNodeExecutor):
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        query_key = str(node.config.get("query_key", "query"))
        query = str(context.get_value(query_key, node.config.get("query", "")))
        output_key = str(node.config.get("output_key", f"{node.id}_result"))
        result = RetrieverFactory.get_hybrid_retriever().retrieve(
            HybridRetrieveQuery(
                query=query,
                knowledge_base_id=node.config.get("knowledge_base_id"),
                top_k=int(node.config.get("top_k", 5)),
                score_threshold=node.config.get("score_threshold"),
                metadata_filter=node.config.get("metadata_filter"),
            )
        )
        result_data = result.model_dump()
        context.set_value(output_key, result_data)
        context.add_artifact(output_key, result_data, {"node_id": node.id, "type": node.type})
        return node.next[0] if node.next else None


class LLMNodeExecutor(BaseWorkflowNodeExecutor):
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        prompt_template = str(node.config.get("prompt", ""))
        output_key = str(node.config.get("output_key", f"{node.id}_result"))
        prompt = _render_template(prompt_template, context.state.values)
        result = LLMFactory.get_llm().chat(
            LLMRequest(messages=[LLMMessage(role="user", content=prompt)])
        )
        result_data = result.model_dump()
        context.set_value(output_key, result_data)
        context.add_artifact(output_key, result_data, {"node_id": node.id, "type": node.type})
        return node.next[0] if node.next else None


class ConditionNodeExecutor(BaseWorkflowNodeExecutor):
    def execute(self, context: WorkflowContext, node: WorkflowNode) -> str | None:
        condition_key = str(node.config.get("condition_key", ""))
        operator = str(node.config.get("operator", "exists"))
        value = context.get_value(condition_key)
        expected = node.config.get("value")
        matched = _evaluate_condition(value=value, operator=operator, expected=expected)
        next_node_id = node.config.get("true_next" if matched else "false_next")
        return str(next_node_id) if next_node_id is not None else None


def get_node_executor(node_type: str) -> BaseWorkflowNodeExecutor:
    executors: dict[str, BaseWorkflowNodeExecutor] = {
        WorkflowNodeType.START: StartNodeExecutor(),
        WorkflowNodeType.END: EndNodeExecutor(),
        WorkflowNodeType.TOOL: ToolNodeExecutor(),
        WorkflowNodeType.RETRIEVER: RetrieverNodeExecutor(),
        WorkflowNodeType.LLM: LLMNodeExecutor(),
        WorkflowNodeType.CONDITION: ConditionNodeExecutor(),
    }
    if node_type not in executors:
        raise ValueError(f"Unsupported workflow node type: {node_type}")
    return executors[node_type]


def _render_template(template: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return str(values.get(key, ""))

    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", replace, template)


def _evaluate_condition(value: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return value is not None
    if operator == "equals":
        return value == expected
    if operator == "not_empty":
        return value is not None and value != "" and value != [] and value != {}
    return False
