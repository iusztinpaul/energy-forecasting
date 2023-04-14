from datetime import datetime
from typing import Optional

import hopsworks
import pandas as pd

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from batch_prediction_pipeline import data
from batch_prediction_pipeline import settings
from batch_prediction_pipeline import utils


logger = utils.get_logger(__name__)


def compute(
    feature_view_version: Optional[int] = None
):
    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]

    logger.info("Loading old predictions...")
    bucket = utils.get_bucket()
    predictions = utils.read_blob_from(
        bucket=bucket,
        blob_name="predictions.parquet"
    )
    predictions.index = predictions.index.set_levels(
        pd.to_datetime(predictions.index.levels[2], unit="h").to_period("H"),
        level=2
    )
    logger.info("Successfully loaded old predictions.")

    logger.info("Connecting to the feature store...")
    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    logger.info("Successfully connected to the feature store.")

    logger.info("Loading latest data from feature store...")
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
    _, latest_observations = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=export_datetime_utc_start,
        end_datetime=export_datetime_utc_end,
    )
    logger.info("Successfully loaded latest data from feature store.")
    
    logger.info("Computing metrics...")
    intersection = pd.merge(
        predictions,
        latest_observations,
        left_index=True,
        right_index=True,
        suffixes=("_predictions", "_latest_observations"),
    )
    if len(intersection) == 0:
        logger.info("Haven't found any new ground truths to compute the metrics on. Exiting...")

        return
    
    mape_metrics = intersection.groupby("datetime_utc").apply(lambda point_in_time: mean_absolute_percentage_error(point_in_time["energy_consumption_latest_observations"], point_in_time["energy_consumption_predictions"], symmetric=False))
    mape_metrics = mape_metrics.rename("MAPE")
    new_metrics = mape_metrics.to_frame()
    logger.info("Successfully computed metrics...")

    logger.info("Saving new metrics...")
    old_metrics = utils.read_blob_from(
        bucket=bucket,
        blob_name="metrics.parquet"
    )
    if old_metrics is None:
        old_metrics = pd.DataFrame(columns=new_metrics.columns)
    old_metrics.index = pd.to_datetime(old_metrics.index, unit="h").to_period("H")
    merged_metrics = new_metrics.merge(
        old_metrics,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_new", "_old"),
    )
    new_metrics = merged_metrics.filter(regex=".*?_new")
    old_metrics = merged_metrics.filter(regex=".*?_old")
    final_metrics = new_metrics.fillna(old_metrics)
    final_metrics.columns = [c.strip("_new") for c in final_metrics.columns]
    final_metrics = final_metrics.sort_index()
    final_metrics = final_metrics.resample("1H").asfreq()
    # Keep only the last 30 days
    final_metrics = final_metrics.tail(n = 24 * 30)
    
    utils.write_blob_to(
        bucket=bucket,
        blob_name="metrics.parquet",
        data=final_metrics
    )
    logger.info("Successfully saved new metrics.")



if __name__ == "__main__":
    compute()
