import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, cast

from backend.app.config.settings import settings
from backend.app.logger import logger
from backend.app.memory.factory import MemoryFactory
from backend.app.tools.base import ToolCall, ToolResult
from backend.app.tools.registry import ToolDisabledError, ToolRegistry, get_tool_registry
from pydantic import ValidationError


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        started_at = datetime.now(UTC)
        logger.info(f"Tool execution started | tool={tool_call.name}")

        try:
            tool = self.registry.get_tool(tool_call.name, require_enabled=True)
        except ToolDisabledError as exc:
            finished_at = datetime.now(UTC)
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=str(exc),
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                    "enabled": False,
                    "registry_version": self.registry.version,
                },
            )
        if tool is None:
            finished_at = datetime.now(UTC)
            return ToolResult(
                name=tool_call.name,
                success=False,
                error="tool not found",
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                    "registry_version": self.registry.version,
                },
            )

        cached_result = self._get_cached_result(tool_call)
        if cached_result is not None:
            return cached_result

        try:
            descriptor = self.registry.get_descriptor(tool_call.name)
            validated_args = tool.args_schema.model_validate(tool_call.arguments)
            result = tool.run(validated_args.model_dump())
            finished_at = datetime.now(UTC)
            duration_ms = _duration_ms(started_at, finished_at)
            result.metadata.update(
                {
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                    "provider": descriptor.provider if descriptor else tool.source,
                    "tool_version": descriptor.version if descriptor else None,
                    "registry_version": self.registry.version,
                    "enabled": descriptor.enabled if descriptor else True,
                    "dynamic_registration": descriptor is not None,
                    "discovery_source": descriptor.provider if descriptor else tool.source,
                }
            )
            self._set_cached_result(tool_call, result)
            logger.info(
                f"Tool execution finished | tool={tool_call.name} | "
                f"success={result.success} | duration_ms={duration_ms}"
            )
            return result
        except ValidationError as exc:
            finished_at = datetime.now(UTC)
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
                    "status": "validation_failed",
                    "reason": "invalid_tool_arguments",
                    "error_type": "tool_validation_error",
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                },
            )
        except Exception as exc:
            finished_at = datetime.now(UTC)
            duration_ms = _duration_ms(started_at, finished_at)
            logger.exception(f"Tool execution failed | tool={tool_call.name}")
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=str(exc),
                metadata={
                    "status": "runtime_failed",
                    "reason": "tool_runtime_error",
                    "error_type": "tool_runtime_error",
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                },
            )

    async def aexecute(self, tool_call: ToolCall) -> ToolResult:
        started_at = datetime.now(UTC)
        logger.info(f"Async tool execution started | tool={tool_call.name}")

        try:
            tool = self.registry.get_tool(tool_call.name, require_enabled=True)
        except ToolDisabledError as exc:
            finished_at = datetime.now(UTC)
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=str(exc),
                metadata={
                    "status": "validation_failed",
                    "reason": "invalid_tool_arguments",
                    "error_type": "tool_validation_error",
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                    "async_execution": True,
                    "sync_fallback": False,
                    "attempt_count": 0,
                    "timeout": False,
                    "cancelled": False,
                    "enabled": False,
                    "registry_version": self.registry.version,
                },
            )
        if tool is None:
            finished_at = datetime.now(UTC)
            return ToolResult(
                name=tool_call.name,
                success=False,
                error="tool not found",
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                    "async_execution": True,
                    "sync_fallback": False,
                    "attempt_count": 0,
                    "timeout": False,
                    "cancelled": False,
                    "registry_version": self.registry.version,
                },
            )

        cached_result = self._get_cached_result(tool_call, async_execution=True)
        if cached_result is not None:
            return cached_result

        try:
            descriptor = self.registry.get_descriptor(tool_call.name)
            validated_args = tool.args_schema.model_validate(tool_call.arguments)
        except ValidationError as exc:
            finished_at = datetime.now(UTC)
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=str(exc),
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                    "async_execution": True,
                    "sync_fallback": False,
                    "attempt_count": 0,
                    "timeout": False,
                    "cancelled": False,
                },
            )

        arguments = validated_args.model_dump()
        max_attempts = max(settings.AGENT_TOOL_RETRY_COUNT, 0) + 1
        sync_fallback = not callable(getattr(tool, "arun", None))
        if sync_fallback and not settings.AGENT_SYNC_FALLBACK_ENABLED:
            finished_at = datetime.now(UTC)
            return ToolResult(
                name=tool_call.name,
                success=False,
                error="sync fallback disabled for tool",
                metadata={
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                    "async_execution": True,
                    "sync_fallback": False,
                    "attempt_count": 0,
                    "timeout": False,
                    "cancelled": False,
                },
            )

        last_error: str | None = None
        retry_errors: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            try:
                async with asyncio.timeout(settings.AGENT_TOOL_TIMEOUT_SECONDS):
                    result = await self._run_tool_async(
                        tool=tool,
                        arguments=arguments,
                        sync_fallback=sync_fallback,
                    )
                finished_at = datetime.now(UTC)
                duration_ms = _duration_ms(started_at, finished_at)
                result.metadata.update(
                    {
                        "started_at": started_at.isoformat(),
                        "finished_at": finished_at.isoformat(),
                        "duration_ms": duration_ms,
                        "async_execution": True,
                        "sync_fallback": sync_fallback,
                        "attempt_count": attempt,
                        "retry_count": attempt - 1,
                        "timeout": False,
                        "cancelled": False,
                        "retry_errors": retry_errors,
                        "provider": descriptor.provider if descriptor else tool.source,
                        "tool_version": descriptor.version if descriptor else None,
                        "registry_version": self.registry.version,
                        "enabled": descriptor.enabled if descriptor else True,
                        "dynamic_registration": descriptor is not None,
                        "discovery_source": descriptor.provider if descriptor else tool.source,
                    }
                )
                self._set_cached_result(tool_call, result)
                logger.info(
                    f"Async tool execution finished | tool={tool_call.name} | "
                    f"success={result.success} | duration_ms={duration_ms}"
                )
                return result
            except TimeoutError:
                last_error = "tool execution timed out"
                retry_errors.append({"attempt": attempt, "error": last_error})
                if attempt >= max_attempts:
                    finished_at = datetime.now(UTC)
                    return ToolResult(
                        name=tool_call.name,
                        success=False,
                        error=last_error,
                        metadata={
                            "status": "runtime_failed",
                            "reason": "tool_runtime_error",
                            "error_type": "tool_runtime_error",
                            "started_at": started_at.isoformat(),
                            "finished_at": finished_at.isoformat(),
                            "duration_ms": _duration_ms(started_at, finished_at),
                            "async_execution": True,
                            "sync_fallback": sync_fallback,
                            "attempt_count": attempt,
                            "retry_count": attempt - 1,
                            "timeout": True,
                            "cancelled": False,
                            "retry_errors": retry_errors,
                        },
                    )
            except asyncio.CancelledError:
                logger.info(f"Async tool execution cancelled | tool={tool_call.name}")
                raise
            except Exception as exc:
                last_error = str(exc)
                retry_errors.append({"attempt": attempt, "error": last_error})
                if attempt >= max_attempts:
                    finished_at = datetime.now(UTC)
                    logger.exception(f"Async tool execution failed | tool={tool_call.name}")
                    return ToolResult(
                        name=tool_call.name,
                        success=False,
                        error=last_error,
                        metadata={
                            "status": "runtime_failed",
                            "reason": "tool_runtime_error",
                            "error_type": "tool_runtime_error",
                            "started_at": started_at.isoformat(),
                            "finished_at": finished_at.isoformat(),
                            "duration_ms": _duration_ms(started_at, finished_at),
                            "async_execution": True,
                            "sync_fallback": sync_fallback,
                            "attempt_count": attempt,
                            "retry_count": attempt - 1,
                            "timeout": False,
                            "cancelled": False,
                            "retry_errors": retry_errors,
                        },
                    )

        finished_at = datetime.now(UTC)
        return ToolResult(
            name=tool_call.name,
            success=False,
            error=last_error or "tool execution failed",
            metadata={
                "status": "runtime_failed",
                "reason": "tool_runtime_error",
                "error_type": "tool_runtime_error",
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_ms": _duration_ms(started_at, finished_at),
                "async_execution": True,
                "sync_fallback": sync_fallback,
                "attempt_count": max_attempts,
                "retry_count": max_attempts - 1,
                "timeout": False,
                "cancelled": False,
                "retry_errors": retry_errors,
            },
        )

    async def _run_tool_async(
        self,
        *,
        tool,
        arguments: dict,
        sync_fallback: bool,
    ) -> ToolResult:
        arun = getattr(tool, "arun", None)
        if callable(arun):
            result = await cast(Awaitable[Any], arun(arguments))
        elif sync_fallback:
            result = await asyncio.to_thread(tool.run, arguments)
        else:
            raise RuntimeError("sync fallback disabled for tool")

        if not isinstance(result, ToolResult):
            raise TypeError("tool returned invalid result")
        return result

    def _get_cached_result(
        self,
        tool_call: ToolCall,
        async_execution: bool = False,
    ) -> ToolResult | None:
        try:
            manager = MemoryFactory.get_manager()
            cached, cache_key = manager.get_tool_cache(tool_call.name, tool_call.arguments)
            if not cached:
                return None
            result = ToolResult.model_validate(cached)
            result.metadata.update(
                {
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "cache_provider": manager.provider.name,
                    "async_execution": async_execution or result.metadata.get(
                        "async_execution", False
                    ),
                }
            )
            return result
        except Exception as exc:
            logger.warning(f"Tool cache lookup failed | tool={tool_call.name} | error={exc}")
            return None

    def _set_cached_result(self, tool_call: ToolCall, result: ToolResult) -> None:
        if not result.success:
            result.metadata.setdefault("cache_hit", False)
            return
        try:
            manager = MemoryFactory.get_manager()
            cache_key = manager.set_tool_cache(
                tool_call.name,
                tool_call.arguments,
                result.model_dump(),
            )
            result.metadata.update(
                {
                    "cache_hit": False,
                    "cache_key": cache_key,
                    "cache_provider": manager.provider.name,
                }
            )
        except Exception as exc:
            result.metadata.setdefault("cache_hit", False)
            result.metadata["cache_error"] = str(exc)
            logger.warning(f"Tool cache write failed | tool={tool_call.name} | error={exc}")


def _duration_ms(started_at: datetime, finished_at: datetime) -> float:
    return round((finished_at - started_at).total_seconds() * 1000, 2)
