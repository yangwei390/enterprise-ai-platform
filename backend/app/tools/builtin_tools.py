import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.app.tools.base import BaseTool, ToolResult


class CurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = "Get the current time as an ISO formatted string."
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": ["string", "null"],
                "description": "IANA timezone name, for example Asia/Shanghai.",
            }
        },
    }

    def run(self, arguments: dict) -> ToolResult:
        timezone = arguments.get("timezone")
        try:
            tz = ZoneInfo(timezone) if timezone else None
        except ZoneInfoNotFoundError:
            return ToolResult(
                name=self.name,
                success=False,
                error=f"Unknown timezone: {timezone}",
            )

        now = datetime.now(tz=tz).isoformat()
        return ToolResult(name=self.name, success=True, result={"time": now})


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely calculate a simple arithmetic expression."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression using numbers and + - * / % ().",
            }
        },
        "required": ["expression"],
    }

    def run(self, arguments: dict) -> ToolResult:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(
                name=self.name,
                success=False,
                error="expression is required",
            )

        value = _safe_eval(expression)
        return ToolResult(name=self.name, success=True, result={"value": value})


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the input text unchanged."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to echo.",
            }
        },
        "required": ["text"],
    }

    def run(self, arguments: dict) -> ToolResult:
        text = arguments.get("text")
        if not isinstance(text, str):
            return ToolResult(name=self.name, success=False, error="text is required")
        return ToolResult(name=self.name, success=True, result={"text": text})


def get_builtin_tools() -> list[BaseTool]:
    return [CurrentTimeTool(), CalculatorTool(), EchoTool()]


def _safe_eval(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        unary_operators = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }
        for operator_type, operation in unary_operators.items():
            if isinstance(node.op, operator_type):
                return operation(operand)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        binary_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }
        for operator_type, operation in binary_operators.items():
            if isinstance(node.op, operator_type):
                return operation(left, right)

    raise ValueError("unsupported expression")
