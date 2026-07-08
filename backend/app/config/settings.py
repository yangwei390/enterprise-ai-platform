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

    LLM_PROVIDER: str = "dummy"
    LLM_MODEL: str = "dummy-llm"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int | None = None
    LLM_TIMEOUT: int = 30
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str | None = None
    LLM_STREAM: bool = False

    MCP_ENABLED: bool = False

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
