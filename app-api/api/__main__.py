import uvicorn

from api.config import get_settings


def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run(
        "api.application:get_app",
        workers=get_settings().WORKERS_COUNT,
        host=get_settings().HOST,
        port=get_settings().PORT,
        reload=get_settings().RELOAD,
        log_level=get_settings().LOG_LEVEL.value.lower(),
        factory=True,
    )


if __name__ == "__main__":
    main()
