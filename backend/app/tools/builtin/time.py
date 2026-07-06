from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.app.tools.base import BaseTool, ToolResult
from backend.app.tools.schemas import CurrentTimeArgs


class CurrentTimeTool(BaseTool):
    name = "get_current_time"
    description = "Get the current time as an ISO formatted string."
    args_schema = CurrentTimeArgs

    def run(self, arguments: dict) -> ToolResult:
        args = CurrentTimeArgs.model_validate(arguments)
        try:
            tz = ZoneInfo(args.timezone) if args.timezone else None
        except ZoneInfoNotFoundError:
            return ToolResult(
                name=self.name,
                success=False,
                error=f"Unknown timezone: {args.timezone}",
            )

        now = datetime.now(tz=tz).isoformat()
        return ToolResult(name=self.name, success=True, result={"time": now})
