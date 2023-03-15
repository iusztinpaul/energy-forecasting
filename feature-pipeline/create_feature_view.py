import datetime
import os

import hopsworks
from dotenv import load_dotenv

import utils

load_dotenv()

FS_API_KEY = os.getenv("FS_API_KEY")


def main():
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()
    energy_consumption_fg = fs.get_feature_group(
        "energy_consumption_denmark", version=1
    )

    # Create feature view.
    ds_query = energy_consumption_fg.select_all()
    feature_view = fs.create_feature_view(
        name="energy_consumption_denmark_view",
        description="Energy consumption for Denmark forecasting model.",
        query=ds_query,
        labels=[]
    )

    # Create the train, validation and test splits.
    # TODO: Find a better way to compute the splits.
    metadata = utils.load_json()
    export_start = datetime.datetime.strptime(metadata["export_datetime_utc_start"], metadata["datetime_format"])
    export_end = datetime.datetime.strptime(metadata["export_datetime_utc_end"], metadata["datetime_format"])
    train_start, train_end = export_start, export_end - datetime.timedelta(
        days=7, minutes=1
    )
    test_start, test_end = export_end - datetime.timedelta(days=7), export_end

    # TODO: Try to make the splits within Hopsworks. But the last time I tried it, it thrown an error.
    feature_view.create_training_data(
        description="Energy consumption training dataset",
        data_format="csv",
        # train_start=train_start,
        # train_end=train_end,
        # test_start=test_start,
        # test_end=test_end,
        write_options={"wait_for_job": False},
        coalesce=True,
    )


if __name__ == "__main__":
    main()
