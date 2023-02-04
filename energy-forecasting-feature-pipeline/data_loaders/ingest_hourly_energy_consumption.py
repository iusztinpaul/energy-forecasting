import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """

    url = 'https://api.energidataservice.dk/dataset/ConsumptionDE35Hour?offset=0&start=2023-01-01T00:00&sort=HourUTC%20DESC&timezone=dk'

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
