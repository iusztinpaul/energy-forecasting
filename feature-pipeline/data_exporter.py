import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FS_API_KEY = os.getenv("FS_API_KEY")


# Connect to feature store.
project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
feature_store = project.get_feature_store()

# Create feature group.
energy_feature_group = feature_store.get_or_create_feature_group(
    name="energy_consumption_denmark",
    version=1,
    description="Denmark hourly energy consumption data. Data is uploaded with an 15 days delay.",
    primary_key=["area", "consumer_type"],
    event_time="datetime_utc",
    online_enabled=False,
)

# Add feature descriptions.
feature_descriptions = [
    {
        "name": "datetime_utc",
        "description": """
                        Datetime interval in UTC when the data was observed.
                        """,
        "validation_rules": "Always full hours, i.e. minutes are 00",
    },
    {
        "name": "area",
        "description": """
                        Denmark is divided in two price areas, divided by the Great Belt: DK1 and DK2.
                        If price area is “DK”, the data covers all Denmark.
                        """,
        "validation_rules": "0 (DK), 1 (DK1) or 2 (Dk2) (int)"
    },
    {
        "name": "consumer_type",
        "description": """
                        The consumer type is the Industry Code DE35 which is owned by Danish Energy. 
                        The code is used by Danish energy companies.
                        """,
        "validation_rules": ">0 (int)"
    },
    {
        "name": "energy_consumption",
        "description": "Total electricity consumption in kWh.",
        "validation_rules": ">=0 (float)"
    },
]
