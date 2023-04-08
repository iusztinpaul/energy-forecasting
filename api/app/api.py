import os
import gcsfs
from typing import Any, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter

from app import schemas 
from app.config import settings

load_dotenv()

GOOGLE_SERVICE_ACCOUNT = os.getenv("GOOGLE_SERVICE_ACCOUNT")
GCP_FILE_PATH = os.getenv("GCP_FILE_PATH")
fs = gcsfs.GCSFileSystem()

api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version="1.0.0"
    )

    return health.dict()

@api_router.get("/consumer_type_values", response_model=schemas.UniqueConsumerType, status_code=200)
def consumer_type_values() -> List:
    
    X = pd.read_parquet(f"{GCP_FILE_PATH}/X.parquet", filesystem=fs)

    unique_consumer_type = list(X.index.unique(level="consumer_type"))

    results = {
        "values": unique_consumer_type
    }
    
    return results

@api_router.get("/area_values", response_model=schemas.UniqueArea, status_code=200)
def area_values() -> List:
    
    X = pd.read_parquet(f"{GCP_FILE_PATH}/X.parquet", filesystem=fs)

    unique_area = list(X.index.unique(level="area"))

    results = {
        "values": unique_area
    }
    
    return results

@api_router.get("/predictions", response_model=schemas.PredictionResults, status_code=200)
def get_predictions() -> Any:
    """
    Get predictions from GCP
    """
    y_train = pd.read_parquet(f"{GCP_FILE_PATH}/y.parquet", filesystem=fs)
    preds = pd.read_parquet(f"{GCP_FILE_PATH}/predictions.parquet", filesystem=fs)
    
    datetime_utc = y_train.index.get_level_values("datetime_utc").to_list()
    area = y_train.index.get_level_values("area").to_list()
    consumer_type = y_train.index.get_level_values("consumer_type").to_list()
    energy_consumption = y_train["energy_consumption"].to_list()

    preds_datetime_utc = preds.index.get_level_values("datetime_utc").to_list() 
    preds_area = preds.index.get_level_values("area").to_list()
    preds_consumer_type = preds.index.get_level_values("consumer_type").to_list()
    preds_energy_consumption = preds["energy_consumption"].to_list()

    results = {
        "datetime_utc": datetime_utc, 
        "area": area,
        "consumer_type": consumer_type,
        "energy_consumption": energy_consumption,
        "preds_datetime_utc": preds_datetime_utc,
        "preds_area": preds_area,
        "preds_consumer_type": preds_consumer_type,
        "preds_energy_consumption": preds_energy_consumption
        }

    return results
