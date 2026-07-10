import json
from pathlib import Path

from backend.app.config.settings import PROJECT_ROOT, settings
from backend.app.logger import logger
from backend.app.mcp.schemas import MCPServerConfig
from backend.app.mcp.security import resolve_env_mapping, validate_server_config


def load_mcp_server_configs(config_path: str | None = None) -> list[MCPServerConfig]:
    path = Path(config_path or settings.MCP_CONFIG_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(f"MCP config loading failed | path={path}")
        raise

    raw_servers = parsed.get("servers", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(raw_servers, list):
        raise ValueError("MCP config must contain a servers list")

    configs: list[MCPServerConfig] = []
    for raw_server in raw_servers:
        if not isinstance(raw_server, dict):
            continue
        server = MCPServerConfig.model_validate(raw_server)
        server = server.model_copy(
            update={
                "headers": resolve_env_mapping(server.headers),
                "env": resolve_env_mapping(server.env),
            }
        )
        validate_server_config(server)
        configs.append(server)
    return configs
