import datetime
import pandas as pd
import requests

from yarl import URL

def load_api_data(*args, **kwargs):
    """
    Template for loading data from API
    """

    url = kwargs.get("api_url", "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour") 
    url = URL(url)
    
    # Data has a delay of 15 days. Thus, we have to shift our window with 15 days.
    days_delay = kwargs.get("days_delay", 15)
    # This is the actual time window of data we will use to train our model on.
    days_export = kwargs.get("days_export", 30)
    # To compute the rolling average and lagged values for all the 30 days of data, 
    # we need to grab some additional days to compute these values for the first elements of the time series.
    # days_rolling_average = kwargs.get("days_rolling_average", [1, 7])
    # n_lagged_days = kwargs.get("n_lagged_days", 3)
    # max_days_rolling_average = max(days_rolling_average)
    # days_extra = max(max_days_rolling_average, n_lagged_days)
    days_extra = 0
    
    current_datetime = datetime.datetime.utcnow()
    export_start = current_datetime - datetime.timedelta(days=days_delay + days_export + days_extra)
    export_start = export_start.strftime("%Y-%m-%dT%H:%M")

    query_params = {
        "offset": 0,
        "sort": "HourUTC",
        "timezone": "utc",
        "start": export_start
    }

    url = url % query_params

    response = requests.get(url)
    response = response.json()

    records = response["records"]
    records = pd.DataFrame.from_records(records)

    return records

