from backend.app.mcp import MCPAdapter, MCPToolConfig
from backend.app.schemas import ApiResponse, success
from backend.app.tools import get_tool_registry
from fastapi import APIRouter

router = APIRouter()


@router.post("/mcp/tools/register", response_model=ApiResponse)
def register_mcp_tool(config: MCPToolConfig) -> ApiResponse:
    tool = MCPAdapter().register_remote_tool(config)
    return success(data=tool.get_definition())


@router.get("/mcp/tools", response_model=ApiResponse)
def list_mcp_tools() -> ApiResponse:
    registry = get_tool_registry()
    tools = [
        tool_definition.model_dump()
        for tool_definition in registry.get_tool_definitions()
        if tool_definition.source in {"mcp", "remote"}
    ]
    return success(data=tools)
