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
    EMBEDDING_BATCH_SIZE: int = 10

    CHUNK_STRATEGY: str = "auto"
    DOCUMENT_CLASSIFICATION_ENABLED: bool = True
    DOCUMENT_STRUCTURE_ENABLED: bool = True
    DOCUMENT_STRUCTURE_FAIL_OPEN: bool = True
    DOCUMENT_STRUCTURE_MAX_NODES: int = 10000
    DOCUMENT_STRUCTURE_MAX_DEPTH: int = 20
    CHUNK_RECURSIVE_SIZE: int = 1000
    CHUNK_RECURSIVE_OVERLAP: int = 150
    CHUNK_MIN_CHARS: int = 100
    CHUNK_LEGAL_MAX_CHARS: int = 1500
    CHUNK_LEGAL_GROUP_SHORT_ARTICLES: bool = True
    CHUNK_LEGAL_MIN_GROUP_CHARS: int = 400
    CHUNK_MARKDOWN_MAX_CHARS: int = 1500
    CHUNK_SEMANTIC_ENABLED: bool = False
    CHUNK_SEMANTIC_SIMILARITY_THRESHOLD: float = 0.55
    CHUNK_SEMANTIC_MAX_PARAGRAPHS: int = 200
    CHUNK_SEMANTIC_BATCH_SIZE: int = 32
    CHUNK_PARENT_CHILD_ENABLED: bool = True
    CHUNK_PARENT_MAX_CHARS: int = 6000
    CHUNK_CHILD_MAX_CHARS: int = 1000
    CHUNK_EMBED_PARENT: bool = False
    STRUCTURE_QUERY_HINT_ENABLED: bool = True
    STRUCTURE_SOFT_BOOST_FACTOR: float = 1.3

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
    EVALUATION_V2_ENABLED: bool = True
    EVALUATION_REPORT_DIR: str = "evaluation/reports"
    EVALUATION_HISTORY_ENABLED: bool = True
    EVALUATION_BASELINE_ENABLED: bool = True
    EVALUATION_MAX_CONCURRENCY: int = 4
    EVALUATION_DEFAULT_TIMEOUT_SECONDS: int = 120
    EVALUATION_FAIL_FAST: bool = False
    EVALUATION_LLM_JUDGE_ENABLED: bool = False
    EVALUATION_LLM_JUDGE_MODEL: str = "qwen-turbo"
    EVALUATION_LLM_JUDGE_TIMEOUT_SECONDS: int = 30
    EVALUATION_LLM_JUDGE_RETRY_COUNT: int = 1
    EVALUATION_COST_CONFIG_PATH: str = "evaluation/v2/costs.yaml"
    EVALUATION_REPORT_MAX_TEXT_CHARS: int = 1000
    EVALUATION_REPORT_REDACT_ENABLED: bool = True

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
    AGENT_LOOP_ENABLED: bool = True
    AGENT_PLANNER_STRATEGY: str = "native_tool_calling"
    AGENT_PLANNER_FALLBACK_ENABLED: bool = True
    AGENT_REFLECTION_ENABLED: bool = True
    AGENT_REFLECTION_AFTER_TOOL_FAILURE: bool = True
    AGENT_REFLECTION_REPEAT_THRESHOLD: int = 2
    AGENT_REFLECTION_MAX_COUNT: int = 2
    AGENT_MAX_STEPS: int = 12
    AGENT_MAX_LLM_CALLS: int = 8
    AGENT_MAX_TOOL_CALLS: int = 10
    AGENT_MAX_REFLECTIONS: int = 2
    AGENT_MAX_SAME_TOOL_REPEATS: int = 2
    AGENT_MAX_DURATION_SECONDS: int = 120
    AGENT_OBSERVATION_MAX_CHARS: int = 6000
    AGENT_MEMORY_MAX_LOOP_MESSAGES: int = 20

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

    WORKFLOW_RUNTIME: str = "v1"
    WORKFLOW_V2_ENABLED: bool = True
    WORKFLOW_DEFAULT_TIMEOUT_SECONDS: int = 120
    WORKFLOW_MAX_STEPS_DEFAULT: int = 20
    WORKFLOW_MAX_CONCURRENCY: int = 4
    WORKFLOW_PARALLEL_FAIL_FAST: bool = False
    WORKFLOW_CHECKPOINT_ENABLED: bool = True
    WORKFLOW_CHECKPOINT_PROVIDER: str = "redis"
    WORKFLOW_CHECKPOINT_FAIL_OPEN: bool = True
    WORKFLOW_CHECKPOINT_TTL_SECONDS: int = 86400
    WORKFLOW_APPROVAL_ENABLED: bool = True
    WORKFLOW_APPROVAL_PERMISSION_ENFORCEMENT: bool = False
    WORKFLOW_APPROVAL_FAIL_OPEN: bool = False
    WORKFLOW_APPROVAL_DEFAULT_TIMEOUT_SECONDS: int = 86400
    WORKFLOW_STREAMING_ENABLED: bool = True
    WORKFLOW_FAIL_OPEN_TO_V1: bool = True

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
