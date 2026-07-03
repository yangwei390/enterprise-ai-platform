import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
    "{file}:{line} | {message}"
)


def setup_logger() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format=LOG_FORMAT)
    logger.add(
        LOG_FILE,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        format=LOG_FORMAT,
    )
