from functools import lru_cache
import logging
import sys
from types import FrameType
from typing import List, Optional, cast

from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # logging levels are type int


class Settings(BaseSettings):
    logging: LoggingSettings = LoggingSettings()

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Energy Consumption API"

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    # e.g: http://localhost,http://localhost:4200,http://localhost:3000
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]

    # Google Cloud Platform credentials
    GCP_PROJECT: Optional[str] = None
    GCP_BUCKET: Optional[str] = None
    GCP_SERVICE_ACCOUNT_JSON_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "APP_API_"
        case_sensitive = False
        env_file_encoding = "utf-8"


# See: https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging  # noqa
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_app_logging(config: Settings) -> None:
    """Prepare custom logging for our application."""

    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )


@lru_cache()
def get_settings():
    return Settings()
