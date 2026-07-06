from datetime import datetime

from backend.app.logger import logger
from backend.app.tools.base import ToolCall, ToolResult
from backend.app.tools.registry import ToolRegistry, get_tool_registry
from pydantic import ValidationError


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        started_at = datetime.utcnow()
        logger.info(f"Tool execution started | tool={tool_call.name}")

        tool = self.registry.get_tool(tool_call.name)
        if tool is None:
            finished_at = datetime.utcnow()
            return ToolResult(
                name=tool_call.name,
                success=False,
                error="tool not found",
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                },
            )

        try:
            validated_args = tool.args_schema.model_validate(tool_call.arguments)
            result = tool.run(validated_args.model_dump())
            finished_at = datetime.utcnow()
            duration_ms = _duration_ms(started_at, finished_at)
            result.metadata.update(
                {
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                }
            )
            logger.info(
                f"Tool execution finished | tool={tool_call.name} | "
                f"success={result.success} | duration_ms={duration_ms}"
            )
            return result
        except ValidationError as exc:
            finished_at = datetime.utcnow()
            duration_ms = _duration_ms(started_at, finished_at)
            logger.info(
                f"Tool argument validation failed | tool={tool_call.name} | "
                f"duration_ms={duration_ms}"
            )
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=str(exc),
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                },
            )
        except Exception as exc:
            finished_at = datetime.utcnow()
            duration_ms = _duration_ms(started_at, finished_at)
            logger.exception(f"Tool execution failed | tool={tool_call.name}")
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=str(exc),
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                },
            )


def _duration_ms(started_at: datetime, finished_at: datetime) -> float:
    return round((finished_at - started_at).total_seconds() * 1000, 2)
