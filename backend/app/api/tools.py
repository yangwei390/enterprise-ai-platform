from backend.app.schemas import ApiResponse, success
from backend.app.tools import ToolCall, ToolExecutor, get_tool_registry
from fastapi import APIRouter

router = APIRouter()


@router.get("/tools", response_model=ApiResponse)
def list_tools() -> ApiResponse:
    registry = get_tool_registry()
    tools = [
        tool_definition.model_dump()
        for tool_definition in registry.get_tool_definitions()
    ]
    return success(data=tools)


@router.post("/tools/execute", response_model=ApiResponse)
def execute_tool(tool_call: ToolCall) -> ApiResponse:
    result = ToolExecutor().execute(tool_call)
    return success(data=result)
