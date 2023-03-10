import os
from pathlib import Path
from typing import Union

import hopsworks
import joblib
import pandas as pd

from dotenv import load_dotenv
from google.cloud import storage
from hsfs.feature_store import FeatureStore

load_dotenv(dotenv_path="../.env.default")
load_dotenv(dotenv_path="../.env", override=True)


WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
FS_API_KEY = os.getenv("FS_API_KEY")


# TODO: Configure fh.
def main(fh: int = 24):
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()

    X, y = load_data_from_feature_store(fs)
    model = load_model_from_model_registry(project)

    predictions = forecast(model, X, fh=fh)

    save(X, y, predictions)
    read()


def load_data_from_feature_store(fs: FeatureStore, target: str = "energy_consumption", feature_view_version: int = 4):
    feature_view = fs.get_feature_view(name="energy_consumption_denmark_view", version=feature_view_version)

    # TODO: Set these days somewhere outside the script.
    current_datetime = pd.Timestamp.now(tz="UTC").replace(minute=0, second=0, microsecond=0)
    # Data is 15 days old, so we always have to shift it with 15 days back as our starting point.
    current_datetime = current_datetime - pd.Timedelta(days=15)
    start_datetime = current_datetime - pd.Timedelta(days=14)

    data = feature_view.get_batch_data(
        start_time=start_datetime, end_time=current_datetime
    )

    # TODO: Can I move the index preparation to the model pipeline?
    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    return X, y


def load_model_from_model_registry(project, model_version: int = 1):
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
    storage_client = storage.Client()

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
    storage_client = storage.Client()

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
    main()
