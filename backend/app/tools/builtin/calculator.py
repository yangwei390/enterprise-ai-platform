import ast
import operator

from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.schemas import CalculatorArgs


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely calculate a simple arithmetic expression."
    args_schema = CalculatorArgs

    def run(self, arguments: dict) -> ToolResult:
        args = CalculatorArgs.model_validate(arguments)
        expression = args.expression.strip()
        if not expression:
            return ToolResult(
                name=self.name,
                success=False,
                error="expression is required",
            )
        if len(expression) > 100:
            return ToolResult(
                name=self.name,
                success=False,
                error="expression is too long",
            )

        value = _safe_eval(expression)
        return ToolResult(name=self.name, success=True, result={"value": value})

    async def arun(self, arguments: dict) -> ToolResult:
        return self.run(arguments)


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
