from datetime import datetime
from pathlib import Path
from typing import Optional

import hopsworks
import pandas as pd

from batch_prediction_pipeline import data
from batch_prediction_pipeline import settings
from batch_prediction_pipeline import utils


logger = utils.get_logger(__name__)


def predict(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    model_version: Optional[int] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
) -> None:
    """Main function used to do batch predictions.

    Args:
        fh (int, optional): forecast horizon. Defaults to 24.
        feature_view_version (Optional[int], optional): feature store feature view version. If None is provided, it will try to load it from the cached feature_view_metadata.json file.
        model_version (Optional[int], optional): model version to load from the model registry. If None is provided, it will try to load it from the cached train_metadata.json file.
        start_datetime (Optional[datetime], optional): start datetime used for extracting features for predictions. If None is provided, it will try to load it from the cached feature_pipeline_metadata.json file.
        end_datetime (Optional[datetime], optional): end datetime used for extracting features for predictions. If None is provided, it will try to load it from the cached feature_pipeline_metadata.json file.
    """

    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]
    if model_version is None:
        train_metadata = utils.load_json("train_metadata.json")
        model_version = train_metadata["model_version"]
    if start_datetime is None or end_datetime is None:
        feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
        start_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
        end_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

    logger.info("Connecting to the feature store...")
    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    logger.info("Successfully connected to the feature store.")

    logger.info("Loading data from feature store...")
    logger.info(f"Loading features from {start_datetime} to {end_datetime}.")
    X, y = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    logger.info("Successfully loaded data from feature store.")

    logger.info("Loading model from model registry...")
    model = load_model_from_model_registry(project, model_version)
    logger.info("Successfully loaded model from model registry.")

    logger.info("Making predictions...")
    predictions = forecast(model, X, fh=fh)
    predictions_start_datetime = predictions.index.get_level_values(
        level="datetime_utc"
    ).min()
    predictions_end_datetime = predictions.index.get_level_values(
        level="datetime_utc"
    ).max()
    logger.info(
        f"Forecasted energy consumption from {predictions_start_datetime} to {predictions_end_datetime}."
    )
    logger.info("Successfully made predictions.")

    logger.info("Saving predictions...")
    save(X, y, predictions)
    logger.info("Successfully saved predictions.")

    # Save the predictions to the bucket for monitoring.
    logger.info("Merging predictions with cached predictions...")
    save_for_monitoring(predictions, start_datetime)
    logger.info("Successfully merged predictions with cached predictions...")


def load_model_from_model_registry(project, model_version: int):
    """
    This function loads a model from the Model Registry.
    The model is downloaded, saved locally, and loaded into memory.
    """

    mr = project.get_model_registry()
    model_registry_reference = mr.get_model(name="best_model", version=model_version)
    model_dir = model_registry_reference.download()
    model_path = Path(model_dir) / "best_model.pkl"

    model = utils.load_model(model_path)

    return model


def forecast(model, X: pd.DataFrame, fh: int = 24):
    """
    Get a forecast of the total load for the given areas and consumer types.

    Args:
        model (sklearn.base.BaseEstimator): Fitted model that implements the predict method.
        X (pd.DataFrame): Exogenous data with area, consumer_type, and datetime_utc as index.
        fh (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast of total load for each area, consumer_type, and datetime_utc.
    """

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
    """Save the input data, target data, and predictions to GCS."""

    # Get the bucket object from the GCS client.
    bucket = utils.get_bucket()

    # Save the input data and target data to the bucket.
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


def save_for_monitoring(predictions: pd.DataFrame, start_datetime: datetime):
    """Save predictions to GCS for monitoring.

    The predictions are saved as a parquet file in GCS.
    The predictions are saved in a bucket with the following structure:
    gs://<BUCKET_NAME>/predictions_monitoring.parquet

    The predictions are stored in a multiindex dataframe with the following indexes:
    - area: The area of the predictions, e.g. "DK1".
    - consumer_type: The consumer type of the predictions, e.g. "residential".
    - datetime_utc: The timestamp of the predictions, e.g. "2020-01-01 00:00:00" with a frequency of 1 hour.
    """

    bucket = utils.get_bucket()

    cached_predictions = utils.read_blob_from(
        bucket=bucket, blob_name=f"predictions_monitoring.parquet"
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
        new_predictions.columns = new_predictions.columns.str.replace("_new", "")
        cached_predictions = merged_predictions.filter(regex=".*?_cached")
        cached_predictions.columns = cached_predictions.columns.str.replace(
            "_cached", ""
        )

        # NOTE: fillna() not working properly on multindex DataFrames. Got nasty bugs because of it.
        new_predictions.update(cached_predictions)
        predictions = new_predictions

    predictions = predictions.loc[
        predictions.index.get_level_values("datetime_utc")
        >= pd.Period(start_datetime, freq="H")
    ]
    predictions = predictions.dropna(subset=["energy_consumption"])

    utils.write_blob_to(
        bucket=bucket,
        blob_name=f"predictions_monitoring.parquet",
        data=predictions,
    )
    logger.info(f"Successfully cached predictions forecasted before {start_datetime}.")


if __name__ == "__main__":
    predict()
