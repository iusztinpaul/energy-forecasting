import os
from pathlib import Path
from typing import Union

import hopsworks
import joblib
import pandas as pd
import wandb

from dotenv import load_dotenv
from google.cloud import storage
from hsfs.feature_store import FeatureStore

load_dotenv(dotenv_path="../.env.default")
load_dotenv(dotenv_path="../.env", override=True)


WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
FS_API_KEY = os.getenv("FS_API_KEY")


def main(fh: int = 24):
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()

    X, y = load_data_from_feature_store(fs)
    model = load_model_from_model_registry(fs)

    predictions = forecast(model, X, fh=fh)

    save(X, y, predictions)
    read()


def load_data_from_feature_store(fs: FeatureStore):
    feature_views = fs.get_feature_views("energy_consumption_denmark_view")
    feature_view = feature_views[-1]
    # TODO: Get the latest training dataset.
    # TODO: Handle hopsworks versions overall.
    X, y = feature_view.get_training_data(training_dataset_version=1)

    # Set the index as is required by sktime.
    data = pd.concat([X, y], axis=1)
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=["energy_consumption"])
    X["area_exog"] = X.index.get_level_values(0)
    X["consumer_type_exog"] = X.index.get_level_values(1)
    # Prepare the time series to be forecasted.
    y = data[["energy_consumption"]]

    return X, y



def load_model_from_model_registry(fs: FeatureStore):
    feature_views = fs.get_feature_views("energy_consumption_denmark_view")
    feature_view = feature_views[-1]

    model_metadata = feature_view.get_training_dataset_tag(name="wandb", training_dataset_version=1)
    model_artifact_name = model_metadata["artifact_name"]

    api = wandb.Api()
    artifact = api.artifact(model_artifact_name)
    download_dir = artifact.download()

    model_path = Path(download_dir) / "best_model.pkl"
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
    all_areas = X.index.get_level_values(level=0).unique()
    all_consumer_types = X.index.get_level_values(level=1).unique()
    latest_datetime = X.index.get_level_values(level=2).max()

    start = latest_datetime + 1
    end = start + fh - 1
    fh_range = pd.date_range(
        start=start.to_timestamp(),
        end=end.to_timestamp(),
        freq="H"
    )
    fh_range = pd.PeriodIndex(fh_range, freq="H")

    index = pd.MultiIndex.from_product(
        [all_areas, all_consumer_types, fh_range],
        names=["area", "consumer_type", "datetime_utc"]
    )
    X_forecast = pd.DataFrame(index=index)
    X_forecast["area_exog"] = X_forecast.index.get_level_values(0)
    X_forecast["consumer_type_exog"] = X_forecast.index.get_level_values(1)

    predictions = model.predict(X=X_forecast)

    return predictions


def save(X: pd.DataFrame, y: pd.DataFrame, predictions: pd.DataFrame):
    storage_client = storage.Client()

    bucket_name = "hourly-batch-predictions"
    bucket = storage_client.bucket(bucket_name=bucket_name)

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
    storage_client = storage.Client()

    bucket_name = "hourly-batch-predictions"
    bucket = storage_client.bucket(bucket_name=bucket_name)

    X_blob = bucket.blob(blob_name="X.parquet")
    with X_blob.open("rb") as f:
        X = pd.read_parquet(f)
        print(X.head())

    y_blob = bucket.blob(blob_name="y.parquet")
    with y_blob.open("rb") as f:
        y = pd.read_parquet(f)
        print(y.head())

    predictions_blob = bucket.blob(blob_name="predictions.parquet")
    with predictions_blob.open("rb") as f:
        predictions = pd.read_parquet(f)
        print(predictions.head())


if __name__ == "__main__":
    main()
