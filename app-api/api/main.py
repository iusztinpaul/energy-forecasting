from typing import Any

import uvicorn
from fastapi import APIRouter, FastAPI

from api.views import api_router
from api.config import get_settings, setup_app_logging

# Setup logging as early as possible.
setup_app_logging(config=get_settings())

app = FastAPI(
    title=get_settings().PROJECT_NAME,
    docs_url=f"{get_settings().API_V1_STR}/docs",
    redoc_url=f"{get_settings().API_V1_STR}/redoc",
    openapi_url=f"{get_settings().API_V1_STR}/openapi.json",
)

root_router = APIRouter()

app.include_router(api_router, prefix=get_settings().API_V1_STR)
app.include_router(root_router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
