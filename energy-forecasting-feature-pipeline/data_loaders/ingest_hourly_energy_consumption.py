import datetime
import pandas as pd
import requests

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

    url = kwargs.get("api_url", "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour") 
    url = URL(url)
    
    days_delay = kwargs.get("days_delay", 15)
    days_export = kwargs.get("days_export", 30)
    current_datetime = datetime.datetime.utcnow()
    export_start = current_datetime - datetime.timedelta(days=days_delay + days_export)
    export_start = export_start.strftime("%Y-%m-%dT%H:%M")

    query_params = {
        "offset": 0,
        "sort": "HourUTC DESC",
        "timezone": "utc",
        "start": export_start
    }
    url = url % query_params

    # url = '?offset=0&start=2023-01-01T00:00&sort=HourUTC%20DESC&timezone=dk'
    print(url)

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
