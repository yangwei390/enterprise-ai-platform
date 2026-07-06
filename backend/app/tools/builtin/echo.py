from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.schemas import EchoArgs


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the input text unchanged."
    args_schema = EchoArgs

    def run(self, arguments: dict) -> ToolResult:
        args = EchoArgs.model_validate(arguments)
        return ToolResult(name=self.name, success=True, result={"text": args.text})
