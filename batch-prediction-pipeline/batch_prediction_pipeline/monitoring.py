from typing import Optional

import hopsworks
import pandas as pd

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from batch_prediction_pipeline import data
from batch_prediction_pipeline import settings
from batch_prediction_pipeline import utils


logger = utils.get_logger(__name__)


def compute(feature_view_version: Optional[int] = None, compute_on_n_days: int = 30):
    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]

    logger.info("Loading old predictions...")
    bucket = utils.get_bucket()
    predictions = utils.read_blob_from(
        bucket=bucket, blob_name=f"predictions_{compute_on_n_days}_days.parquet"
    )
    if predictions is None:
        logger.info(
            "Haven't found any predictions to compute the metrics on. Exiting..."
        )

        return
    predictions.index = predictions.index.set_levels(
        pd.to_datetime(predictions.index.levels[2], unit="h").to_period("H"), level=2
    )
    logger.info("Successfully loaded old predictions.")

    logger.info("Connecting to the feature store...")
    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    logger.info("Successfully connected to the feature store.")

    logger.info("Loading latest data from feature store...")
    predictions_min_datetime_utc = (
        predictions.index.get_level_values("datetime_utc").min().to_timestamp()
    )
    predictions_max_datetime_utc = (
        predictions.index.get_level_values("datetime_utc").max().to_timestamp()
    )
    _, latest_observations = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=predictions_min_datetime_utc,
        end_datetime=predictions_max_datetime_utc,
    )
    logger.info("Successfully loaded latest data from feature store.")

    if len(latest_observations) == 0:
        logger.info(
            "Haven't found any new ground truths to compute the metrics on. Exiting..."
        )

        return

    logger.info("Computing metrics...")
    predictions["energy_consumption_observations"] = latest_observations[
        "energy_consumption"
    ]
    predictions = predictions.rename(
        columns={"energy_consumption": "energy_consumption_predictions"}
    )
    # Compute metrics only on data points that have ground truth.
    predictions = predictions.dropna(subset=["energy_consumption_observations"])

    mape_metrics = predictions.groupby("datetime_utc").apply(
        lambda point_in_time: mean_absolute_percentage_error(
            point_in_time["energy_consumption_observations"],
            point_in_time["energy_consumption_predictions"],
            symmetric=False,
        )
    )
    mape_metrics = mape_metrics.rename("MAPE")
    metrics = mape_metrics.to_frame()
    logger.info("Successfully computed metrics...")

    logger.info("Saving new metrics...")
    utils.write_blob_to(bucket=bucket, blob_name=f"metrics_{compute_on_n_days}_days.parquet", data=metrics)
    utils.write_blob_to(bucket=bucket, blob_name=f"y_{compute_on_n_days}_days.parquet", data=latest_observations[["energy_consumption"]])
    logger.info("Successfully saved new metrics.")


if __name__ == "__main__":
    compute()
