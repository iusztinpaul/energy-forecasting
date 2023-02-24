import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FS_API_KEY = os.getenv("FS_API_KEY")


def export_to_feature_store(filepath):
    # Read data.
    energy_consumption_data = pd.read_parquet(filepath)
    # energy_consumption_data["datetime_utc"] = pd.to_datetime(
    #     energy_consumption_data["datetime_utc"]
    # )

    # Connect to feature store.
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    feature_store = project.get_feature_store()

    # Create feature group.
    energy_feature_group = feature_store.get_or_create_feature_group(
        name="energy_consumption_batch",
        version=1,
        description="Denmark hourly energy consumption data. Data is uploaded with an 15 days delay.",
        primary_key=["area", "consumer_type"],
        event_time="datetime_utc",
        online_enabled=True,
    )
    energy_feature_group.insert(energy_consumption_data)

    # Add feature descriptions.
    feature_descriptions = [
        {
            "name": "datetime_utc",
            "description": """
                            Datetime in UTC when the data was observed. A date and time (interval), 
                            shown in UTC time zone, where the values are valid. 00:00 o’clock is the first hour of a 
                            given day interval 00:00 - 00:59 and 01:00 covers the second hour (interval) of the day 
                            and so forth. Please note: The naming is based on the length of the interval of the 
                            finest grain of the resolution.
                           """,
            "validation_rules": "Always full hours, i.e. minutes are 00",
        },
        {
            "name": "area",
            "description": """
                            Denmark is divided in two price areas, or bidding zones, divided by the Great Belt. 
                            DK1 is west of the Great Belt and DK2 is east of the Great Belt. If price area is “DK”, 
                            the data covers all Denmark.
                           """,
            "validation_rules": "0 (DK), 1 (DK1) or 2 (Dk2) (int)"
        },
        {
            "name": "consumer_type",
            "description": """
                            The consumer type is the Industry Code DE35 which is owned and maintained by Danish Energy, 
                            a non-commercial lobby organization for Danish energy compa-nies. 
                            The code is used by Danish energy companies.
                           """,
            "validation_rules": ">0 (int)"
        },
        {
            "name": "consumption",
            "description": "Total electricity consumption in kWh.",
            "validation_rules": ">=0 (float)"
        },
    ]
    for description in feature_descriptions:
        energy_feature_group.update_feature_description(
            description["name"], description["description"]
        )


if __name__ == "__main__":
    export_to_feature_store("energy_consumption_data.parquet")
