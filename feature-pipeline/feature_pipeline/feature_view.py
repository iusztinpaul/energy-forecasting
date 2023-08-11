from datetime import datetime
from typing import Optional

import fire
import hopsworks

from feature_pipeline import utils
from feature_pipeline import settings
import hsfs


logger = utils.get_logger(__name__)


def create(
    feature_group_version: Optional[int] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
) -> dict:
    """Create a new feature view version and training dataset
    based on the given feature group version and start and end datetimes.

    Args:
        feature_group_version (Optional[int]): The version of the
            feature group. If None is provided, it will try to load it
            from the cached feature_pipeline_metadata.json file.
        start_datetime (Optional[datetime]): The start
            datetime of the training dataset that will be created.
            If None is provided, it will try to load it
            from the cached feature_pipeline_metadata.json file.
        end_datetime (Optional[datetime]): The end
            datetime of the training dataset that will be created.
              If None is provided, it will try to load it
            from the cached feature_pipeline_metadata.json file.

    Returns:
        dict: The feature group version.

    """

    if feature_group_version is None:
        feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
        feature_group_version = feature_pipeline_metadata["feature_group_version"]

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

    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"],
        project=settings.SETTINGS["FS_PROJECT_NAME"],
    )
    fs = project.get_feature_store()

    # Delete old feature views as the free tier only allows 100 feature views.
    # NOTE: Normally you would not want to delete feature views. We do it here just to stay in the free tier.
    try:
        feature_views = fs.get_feature_views(name="energy_consumption_denmark_view")
    except hsfs.client.exceptions.RestAPIError:
        logger.info("No feature views found for energy_consumption_denmark_view.")

        feature_views = []

    for feature_view in feature_views:
        try:
            feature_view.delete_all_training_datasets()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete training datasets for feature view {feature_view.name} with version {feature_view.version}."
            )

        try:
            feature_view.delete()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete feature view {feature_view.name} with version {feature_view.version}."
            )

    # Create feature view in the given feature group version.
    energy_consumption_fg = fs.get_feature_group(
        "energy_consumption_denmark", version=feature_group_version
    )
    ds_query = energy_consumption_fg.select_all()
    feature_view = fs.create_feature_view(
        name="energy_consumption_denmark_view",
        description="Energy consumption for Denmark forecasting model.",
        query=ds_query,
        labels=[],
    )

    # Create training dataset.
    logger.info(
        f"Creating training dataset between {start_datetime} and {end_datetime}."
    )
    feature_view.create_training_data(
        description="Energy consumption training dataset",
        data_format="csv",
        start_time=start_datetime,
        end_time=end_datetime,
        write_options={"wait_for_job": True},
        coalesce=False,
    )

    # Save metadata.
    metadata = {
        "feature_view_version": feature_view.version,
        "training_dataset_version": 1,
    }
    utils.save_json(
        metadata,
        file_name="feature_view_metadata.json",
    )

    return metadata


if __name__ == "__main__":
    fire.Fire(create)
