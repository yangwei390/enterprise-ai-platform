from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    APP_NAME: str
    APP_ENV: str
    APP_HOST: str
    APP_PORT: int
    APP_VERSION: str

    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str | None = None
    REDIS_ENABLED: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PREFIX: str = "enterprise_ai"
    REDIS_SESSION_TTL_SECONDS: int = 86400
    REDIS_TOOL_CACHE_TTL_SECONDS: int = 3600
    REDIS_CHECKPOINT_TTL_SECONDS: int = 86400

    QDRANT_HOST: str
    QDRANT_PORT: int

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = ""

    EMBEDDING_MODEL_PATH: str = ""
    RERANKER_MODEL_PATH: str = ""

    EMBEDDING_PROVIDER: str = "dummy"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_BASE_URL: str | None = None
    EMBEDDING_DIMENSION: int | None = None

    RERANK_PROVIDER: str = "dummy"
    RERANK_MODEL: str = "gte-rerank-v2"
    RERANK_API_KEY: str | None = None
    RERANK_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    RERANK_TOP_K: int = 20
    RERANK_TIMEOUT: int = 30
    RERANK_FAIL_OPEN: bool = True

    CONTEXT_COMPRESSION_ENABLED: bool = True
    CONTEXT_COMPRESSION_PROVIDER: str = "rule_based"
    CONTEXT_COMPRESSION_MAX_CHARS: int = 6000
    CONTEXT_COMPRESSION_MAX_CHUNK_CHARS: int = 1200
    CONTEXT_COMPRESSION_FAIL_OPEN: bool = True
    CONTEXT_COMPRESSION_LLM_MODEL: str = "qwen-turbo"
    CONTEXT_COMPRESSION_LLM_TEMPERATURE: float = 0
    CONTEXT_COMPRESSION_LLM_MAX_CHARS_PER_CHUNK: int = 1200
    CONTEXT_COMPRESSION_LLM_TIMEOUT_SECONDS: int = 30
    CONTEXT_COMPRESSION_LLM_MAX_CALLS: int = 8

    NEIGHBOR_EXPANSION_ENABLED: bool = True
    NEIGHBOR_EXPANSION_BEFORE: int = 1
    NEIGHBOR_EXPANSION_AFTER: int = 1
    NEIGHBOR_EXPANSION_MAX_ADDED_CHUNKS: int = 10
    NEIGHBOR_EXPANSION_FAIL_OPEN: bool = True

    MMR_ENABLED: bool = True
    MMR_LAMBDA: float = 0.7
    MMR_TOP_K: int = 5
    MMR_MIN_SCORE: float = 0.0
    MMR_FAIL_OPEN: bool = True
    MMR_SIMILARITY_THRESHOLD: float = 0.85

    EVALUATION_KEYWORD_THRESHOLD: float = 0.5

    LLM_PROVIDER: str = "dummy"
    LLM_MODEL: str = "dummy-llm"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int | None = None
    LLM_TIMEOUT: int = 30
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str | None = None
    LLM_STREAM: bool = False

    MCP_ENABLED: bool = False
    MCP_CONFIG_PATH: str = "config/mcp/servers.json"
    MCP_DISCOVERY_ON_STARTUP: bool = True
    MCP_FAIL_OPEN: bool = True
    MCP_DEFAULT_CONNECT_TIMEOUT_SECONDS: int = 10
    MCP_DEFAULT_TOOL_TIMEOUT_SECONDS: int = 30
    MCP_DEFAULT_RETRY_COUNT: int = 1
    MCP_MAX_CONCURRENCY: int = 4
    MCP_STDIO_ENABLED: bool = True
    MCP_STREAMABLE_HTTP_ENABLED: bool = True
    MCP_SSE_COMPAT_ENABLED: bool = False
    MCP_AUTO_RECONNECT: bool = True
    MCP_HEALTH_CHECK_ENABLED: bool = True
    MCP_HEALTH_CHECK_INTERVAL_SECONDS: int = 60
    MCP_AUDIT_ENABLED: bool = True
    MCP_ALLOW_INSECURE_HTTP_LOCALHOST: bool = True
    MCP_STDIO_ALLOWED_COMMANDS: str = "python,python3,python3.12,node,npx,uv,uvx"
    MCP_PERMISSION_ENFORCEMENT_ENABLED: bool = False
    MCP_PERMISSION_FAIL_OPEN: bool = False

    MEMORY_PROVIDER: str = "redis"

    AGENT_RUNTIME: str = "v1"
    AGENT_ASYNC_ENABLED: bool = True
    AGENT_ASYNC_TIMEOUT_SECONDS: int = 60
    AGENT_TOOL_TIMEOUT_SECONDS: int = 30
    AGENT_TOOL_MAX_CONCURRENCY: int = 4
    AGENT_TOOL_RETRY_COUNT: int = 1
    AGENT_ASYNC_FAIL_OPEN: bool = True
    AGENT_SYNC_FALLBACK_ENABLED: bool = True

    DYNAMIC_TOOL_REGISTRY_ENABLED: bool = True
    TOOL_REGISTRY_AUTO_REFRESH: bool = False
    TOOL_REGISTRY_REFRESH_INTERVAL_SECONDS: int = 300
    TOOL_REGISTRY_FAIL_OPEN: bool = True

    TOOL_PLUGIN_ENABLED: bool = True
    TOOL_PLUGIN_PATH: str = "backend/app/tools/plugins"

    HTTP_TOOL_PROVIDER_ENABLED: bool = False
    HTTP_TOOL_CONFIG_PATH: str = "config/tools/http_tools.json"
    HTTP_TOOL_DEFAULT_TIMEOUT_SECONDS: int = 30

    WORKFLOW_TOOL_PROVIDER_ENABLED: bool = True
    MCP_TOOL_PROVIDER_ENABLED: bool = False

    UPLOAD_DIR: str

    @field_validator("LLM_MAX_TOKENS", "EMBEDDING_DIMENSION", mode="before")
    @classmethod
    def parse_optional_int(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator(
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "RERANK_API_KEY",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, value: object) -> object:
        if value == "":
            return None
        return value

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


class ConfigResponse(BaseModel):
    app_name: str
    app_env: str
    app_version: str
    app_host: str
    app_port: int
