import datetime
import pandas as pd
import requests

from yarl import URL


def from_api(**kwargs):
    """
    Template for loading data from API
    """

    url = kwargs.get(
        "api_url", "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour"
    )
    url = URL(url)

    # Data has a delay of 15 days. Thus, we have to shift our window with 15 days.
    days_delay = kwargs.get("days_delay", 15)
    # This is the actual time window of data we will use to train our model on.
    days_export = kwargs.get("days_export", 30)

    current_datetime = datetime.datetime.utcnow()
    export_start = current_datetime - datetime.timedelta(days=days_delay + days_export)
    export_start = export_start.strftime("%Y-%m-%dT%H:%M")

    query_params = {
        "offset": 0,
        "sort": "HourUTC",
        "timezone": "utc",
        "start": export_start,
    }

    url = url % query_params

    response = requests.get(url)
    response = response.json()

    records = response["records"]
    records = pd.DataFrame.from_records(records)

    return records
