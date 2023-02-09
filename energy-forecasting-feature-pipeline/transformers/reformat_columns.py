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
        "HourUTC": "UTCDatetime",
        "PriceArea": "Area",
        "ConsumerType_DE35": "ConsumerType",
        "TotalCon": "EnergyConsumption"
    })
    df = df.drop(columns=["HourDK"])

    # Casting
    df["UTCDatetime"] = pd.to_datetime(df["UTCDatetime"])
    df["Area"] = df["Area"].astype("string")
    df["ConsumerType"] = df["ConsumerType"].astype("string")
    df["EnergyConsumption"] = df["EnergyConsumption"].astype("float64")

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
