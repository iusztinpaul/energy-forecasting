import enum
from functools import lru_cache
import logging
import sys
from types import FrameType
from typing import List, Optional, cast

from pydantic import AnyHttpUrl, BaseSettings


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # General configurations.
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: LogLevel = LogLevel.INFO
    # - Current version of the API.
    VERSION: str = "v1"
    # - Quantity of workers for uvicorn.
    WORKERS_COUNT: int = 1
    # - Enable uvicorn reloading.
    RELOAD: bool = False

    PROJECT_NAME: str = "Energy Consumption API"

    # Google Cloud Platform credentials
    GCP_PROJECT: Optional[str] = None
    GCP_BUCKET: Optional[str] = None
    GCP_SERVICE_ACCOUNT_JSON_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "APP_API_"
        case_sensitive = False
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
