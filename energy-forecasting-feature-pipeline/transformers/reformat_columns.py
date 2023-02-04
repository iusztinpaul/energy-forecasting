import pandas as pd

from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def execute_transformer_action(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Execute Transformer Action: ActionType.REFORMAT

    Docs: https://docs.mage.ai/guides/transformer-blocks#reformat-values
    """

    # Renaming
    df = df.rename(columns={
        "HourUTC": "UTC Datetime",
        "PriceArea": "Area",
        "ConsumerType_DE35": "Consumer Type",
        "TotalCon": "Energy Consumption"
    })
    df = df.drop(columns=["HourDK"])

    # Casting
    df["UTC Datetime"] = pd.to_datetime(df["UTC Datetime"])
    df["Area"] = df["Area"].astype("string")
    df["Consumer Type"] = df["Consumer Type"].astype("string")
    df["Energy Consumption"] = df["Energy Consumption"].astype("float64")

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
