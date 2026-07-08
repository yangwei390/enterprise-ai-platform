from backend.app.context_compression.base import BaseContextCompressor
from backend.app.context_compression.config import get_context_compression_config
from backend.app.context_compression.rule_based_compressor import (
    RuleBasedContextCompressor,
)
from backend.app.logger import logger


class ContextCompressorFactory:
    @staticmethod
    def get_compressor(provider: str | None = None) -> BaseContextCompressor:
        config = get_context_compression_config()
        selected_provider = (provider or config.provider).lower()

        if selected_provider == "rule_based":
            return RuleBasedContextCompressor(max_chunk_chars=config.max_chunk_chars)

        logger.warning(
            "Unknown context compression provider, fallback to rule_based: {}",
            selected_provider,
        )
        return RuleBasedContextCompressor(max_chunk_chars=config.max_chunk_chars)
