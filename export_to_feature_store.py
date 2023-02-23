import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FS_API_KEY")


def export_to_feature_store(filepath):
    # Read data
    energy_consumption_data = pd.read_parquet(filepath)
    energy_consumption_data["datetime_utc"] = pd.to_datetime(
        energy_consumption_data["datetime_utc"]
    )
    # TODO: Move this cleaning to DE pipeline
    energy_consumption_data["area"] = pd.Categorical(
        energy_consumption_data["area"],
        categories=energy_consumption_data["area"].unique()
    ).codes
    energy_consumption_data["consumer_type"] = energy_consumption_data["consumer_type"].astype("int64")

    # Connect to feature store
    project = hopsworks.login(api_key_value=API_KEY, project="energy_consumption")
    feature_store = project.get_feature_store()


    # TODO: Write features descriptions after I clean the DE pipeline

    # Create feature group
    energy_feature_group = feature_store.get_or_create_feature_group(
        name="energy_consumption",
        version=1,
        description="Inital feature group",
        primary_key=["datetime_utc", "area", "consumer_type"],
    )
    energy_feature_group.insert(energy_consumption_data)


if __name__ == "__main__":
    export_to_feature_store("energy_consumption_data.parquet")
