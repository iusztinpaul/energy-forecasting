import datetime
import pandas as pd
import requests
import logging

from yarl import URL

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    
    logging.basicConfig(level=logging.DEBUG)

    url = kwargs.get("api_url", "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour") 
    url = URL(url)
    
    # Data has a delay of 15 days. Thus, we have to shift our window with 15 days.
    days_delay = kwargs.get("days_delay", 15)
    # This is the actual time window of data we will use to train our model on.
    days_export = kwargs.get("days_export", 30)
    # To compute the rolling average and lagged values for all the 30 days of data, 
    # we need to grab some additional days to compute these values for the first elements of the time series.
    days_rolling_average = kwargs.get("days_rolling_average", 1)
    n_lagged_days = kwargs.get("n_lagged_days", 3)
    days_extra = max(days_rolling_average, n_lagged_days)
    
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

    # TODO: Fix the logger. See why it is not working.
    logging.info(f"Calling {url}")

    response = requests.get(url)
    response = response.json()

    records = response["records"]
    records = pd.DataFrame.from_records(records)

    return records


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
