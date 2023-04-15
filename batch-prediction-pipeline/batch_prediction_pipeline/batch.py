from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import hopsworks
import joblib
import pandas as pd

from batch_prediction_pipeline import data
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
    # TODO: Parameterize start and end datetime
    feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
    export_datetime_utc_start = datetime.strptime(
        feature_pipeline_metadata["export_datetime_utc_start"],
        feature_pipeline_metadata["datetime_format"],
    )
    export_datetime_utc_end = datetime.strptime(
        feature_pipeline_metadata["export_datetime_utc_end"],
        feature_pipeline_metadata["datetime_format"],
    )
    X, y = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=export_datetime_utc_start,
        end_datetime=export_datetime_utc_end,
    )
    logger.info("Successfully loaded data from feature store.")

    logger.info("Loading model from model registry...")
    model = load_model_from_model_registry(project, model_version)
    logger.info("Successfully loaded model from model registry.")

    logger.info("Making predictions...")
    predictions = forecast(model, X, fh=fh)
    logger.info("Successfully made predictions.")

    logger.info("Saving predictions...")
    save(X, y, predictions)
    logger.info("Successfully saved predictions.")


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
    bucket = utils.get_bucket()

    for df, blob_name in zip(
        [X, y, predictions], ["X.parquet", "y.parquet", "predictions.parquet"]
    ):
        logger.info(f"Saving {blob_name} to bucket...")
        utils.write_blob_to(
            bucket=bucket,
            blob_name=blob_name,
            data=df,
        )
        logger.info(f"Successfully saved {blob_name} to bucket.")

    logger.info("Merging predictions with cached predictions...")
    merge(predictions)
    logger.info("Successfully merged predictions with cached predictions...")


def merge(predictions: pd.DataFrame, keep_n_days: int = 30):
    bucket = utils.get_bucket()

    cached_predictions = utils.read_blob_from(
        bucket=bucket, blob_name=f"predictions_{keep_n_days}_days.parquet"
    )
    has_cached_predictions = cached_predictions is not None
    if has_cached_predictions is True:
        # Merge predictions with cached predictions.
        cached_predictions.index = cached_predictions.index.set_levels(
            pd.to_datetime(cached_predictions.index.levels[2], unit="h").to_period("H"),
            level=2,
        )

        merged_predictions = predictions.merge(
            cached_predictions,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_new", "_cached"),
        )
        new_predictions = merged_predictions.filter(regex=".*?_new")
        cached_predictions = merged_predictions.filter(regex=".*?_cached")
        predictions = new_predictions.fillna(cached_predictions)
        predictions.columns = predictions.columns.str.replace("_new", "")

    # Make sure that the predictions are sorted and continous.
    predictions = predictions.sort_index()
    predictions = (
        predictions
        .unstack(level=[0, 1])
        .resample("1H")
        .asfreq()
        .stack(level=[2, 1])
        .swaplevel(2, 0)
    )
    # Keep only the last n_days of observations + 1 day of predictions.
    count_unique_area_types = len(predictions.index.get_level_values(level=0).unique())
    count_unique_consumer_types = len(
        predictions.index.get_level_values(level=1).unique()
    )
    predictions = predictions.tail(
        n=(keep_n_days + 1) * 24 * count_unique_area_types * count_unique_consumer_types
    )

    utils.write_blob_to(
        bucket=bucket,
        blob_name=f"predictions_{keep_n_days}_days.parquet",
        data=predictions,
    )


if __name__ == "__main__":
    predict()
