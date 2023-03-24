import datetime

import fire
import hopsworks

from feature_pipeline import utils
from feature_pipeline import settings


def run(feature_group_version: int = 1):
    project = hopsworks.login(
        api_key_value=settings.CREDENTIALS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    energy_consumption_fg = fs.get_feature_group(
        "energy_consumption_denmark", version=feature_group_version
    )

    # Create feature view.
    # TODO: Can I put the [start, end] interval also in the create_feature_view() method?
    metadata = utils.load_json(file_name="feature_pipeline_metadata.json")
    export_start = datetime.datetime.strptime(
        metadata["export_datetime_utc_start"], metadata["datetime_format"]
    )
    export_end = datetime.datetime.strptime(
        metadata["export_datetime_utc_end"], metadata["datetime_format"]
    )

    ds_query = energy_consumption_fg.select_all()
    feature_view = fs.create_feature_view(
        name="energy_consumption_denmark_view",
        description="Energy consumption for Denmark forecasting model.",
        query=ds_query,
        labels=[],
    )

    # Create training dataset.
    # TODO: Try to make the splits within Hopsworks. But the last time I tried it, it thrown an error.
    feature_view.create_training_data(
        description="Energy consumption training dataset",
        data_format="csv",
        start_time=export_start,
        end_time=export_end,
        write_options={"wait_for_job": True},
        coalesce=True,
    )

    utils.save_json(
        {"feature_view_version": feature_view.version, "training_dataset_version": 1},
        file_name="feature_view_metadata.json",
    )


if __name__ == "__main__":
    fire.Fire(run)
