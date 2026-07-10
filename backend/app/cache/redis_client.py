from backend.app.config.settings import settings
from redis import Redis

_redis_client: Redis | None = None


def get_redis_client() -> Redis:
    global _redis_client

    if _redis_client is None:
        if settings.REDIS_URL:
            _redis_client = Redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True,
            )
        else:
            _redis_client = Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True,
            )

    return _redis_client


def close_redis_client() -> None:
    global _redis_client

    if _redis_client is not None:
        _redis_client.close()
        _redis_client = None
