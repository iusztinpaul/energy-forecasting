from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import hopsworks
import joblib
import pandas as pd

from google.cloud import storage
from hsfs.feature_store import FeatureStore

from batch_prediction_pipeline import settings
from batch_prediction_pipeline import utils


logger = utils.get_logger(__name__)


def predict(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    model_version: Optional[int] = None,
):
    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]
    if model_version is None:
        train_metadata = utils.load_json("train_metadata.json")
        model_version = train_metadata["model_version"]

    logger.info("Connecting to the feature store...")
    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    logger.info("Successfully connected to the feature store.")

    logger.info("Loading data from feature store...")
    feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
    export_datetime_utc_start = datetime.strptime(
        feature_pipeline_metadata["export_datetime_utc_start"],
        feature_pipeline_metadata["datetime_format"],
    )
    export_datetime_utc_end = datetime.strptime(
        feature_pipeline_metadata["export_datetime_utc_end"],
        feature_pipeline_metadata["datetime_format"],
    )
    X, y = load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=export_datetime_utc_start,
        end_datetime=export_datetime_utc_end,
    )
    logger.info("Successfully loaded data from feature store.")

    # TODO: Compute metrics on new data.

    logger.info("Loading model from model registry...")
    model = load_model_from_model_registry(project, model_version)
    logger.info("Successfully loaded model from model registry.")

    logger.info("Making predictions...")
    predictions = forecast(model, X, fh=fh)
    logger.info("Successfully made predictions.")

    logger.info("Saving predictions...")
    save(X, y, predictions)
    logger.info("Successfully saved predictions.")

    read()


def load_data_from_feature_store(
    fs: FeatureStore,
    feature_view_version: int,
    start_datetime: datetime,
    end_datetime: datetime,
    target: str = "energy_consumption",
):
    feature_view = fs.get_feature_view(
        name="energy_consumption_denmark_view", version=feature_view_version
    )

    data = feature_view.get_batch_data(start_time=start_datetime, end_time=end_datetime)

    # TODO: Can I move the index preparation to the model pipeline?
    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    return X, y


def load_model_from_model_registry(project, model_version: int):
    mr = project.get_model_registry()
    model_registry_reference = mr.get_model(name="best_model", version=model_version)
    model_dir = model_registry_reference.download()
    model_path = Path(model_dir) / "best_model.pkl"

    model = load_model(model_path)

    return model


def load_model(model_path: Union[str, Path]):
    """
    Template for loading a model.

    Args:
        model_path: Path to the model.

    Returns: Loaded model.
    """

    return joblib.load(model_path)


def forecast(model, X: pd.DataFrame, fh: int = 24):
    # TODO: Make this function independent of X.
    all_areas = X.index.get_level_values(level=0).unique()
    all_consumer_types = X.index.get_level_values(level=1).unique()
    latest_datetime = X.index.get_level_values(level=2).max()

    start = latest_datetime + 1
    end = start + fh - 1
    fh_range = pd.date_range(
        start=start.to_timestamp(), end=end.to_timestamp(), freq="H"
    )
    fh_range = pd.PeriodIndex(fh_range, freq="H")

    index = pd.MultiIndex.from_product(
        [all_areas, all_consumer_types, fh_range],
        names=["area", "consumer_type", "datetime_utc"],
    )
    X_forecast = pd.DataFrame(index=index)
    X_forecast["area_exog"] = X_forecast.index.get_level_values(0)
    X_forecast["consumer_type_exog"] = X_forecast.index.get_level_values(1)

    predictions = model.predict(X=X_forecast)

    return predictions


def save(X: pd.DataFrame, y: pd.DataFrame, predictions: pd.DataFrame):
    storage_client = storage.Client.from_service_account_json(
        json_credentials_path=settings.SETTINGS[
            "GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"
        ],
        project=settings.SETTINGS["GOOGLE_CLOUD_PROJECT"],
    )

    bucket_name = "hourly-batch-predictions"
    bucket = storage_client.bucket(bucket_name=bucket_name)

    # TODO: Standardize to blob process.
    X_blob = bucket.blob(blob_name="X.parquet")
    with X_blob.open("wb") as f:
        X.to_parquet(f)

    y_blob = bucket.blob(blob_name="y.parquet")
    with y_blob.open("wb") as f:
        y.to_parquet(f)

    predictions_blob = bucket.blob(blob_name="predictions.parquet")
    with predictions_blob.open("wb") as f:
        predictions.to_parquet(f)


def read():
    # TODO: Delete this function.
    storage_client = storage.Client.from_service_account_json(
        json_credentials_path=settings.SETTINGS[
            "GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"
        ],
        project=settings.SETTINGS["GOOGLE_CLOUD_PROJECT"],
    )

    bucket_name = "hourly-batch-predictions"
    bucket = storage_client.bucket(bucket_name=bucket_name)

    X_blob = bucket.blob(blob_name="X.parquet")
    with X_blob.open("rb") as f:
        X = pd.read_parquet(f)
        print("X:")
        print(X.head())

    y_blob = bucket.blob(blob_name="y.parquet")
    with y_blob.open("rb") as f:
        y = pd.read_parquet(f)
        print("y:")
        print(y.head())

    predictions_blob = bucket.blob(blob_name="predictions.parquet")
    with predictions_blob.open("rb") as f:
        predictions = pd.read_parquet(f)
        print("predictions:")
        print(predictions.head())


if __name__ == "__main__":
    predict()
