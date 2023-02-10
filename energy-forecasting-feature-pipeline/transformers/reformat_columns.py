import pandas as pd

from mage_ai.data_cleaner.transformer_actions.base import BaseAction
from mage_ai.data_cleaner.transformer_actions.constants import ActionType, Axis
from mage_ai.data_cleaner.transformer_actions.utils import build_transformer_action
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
        "HourUTC": "DatetimeUtc",
        "PriceArea": "Area",
        "ConsumerType_DE35": "ConsumerType",
        "TotalCon": "EnergyConsumption"
    })
    df = df.drop(columns=["HourDK"])

    action = build_transformer_action(
        df,
        action_type=ActionType.CLEAN_COLUMN_NAME,
        arguments=df.columns,
        axis=Axis.COLUMN,
    )
    df = BaseAction(action).execute(df)

    # Casting
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["area"] = df["area"].astype("string")
    df["consumer_type"] = df["consumer_type"].astype("string")
    df["energy_consumption"] = df["energy_consumption"].astype("float64")

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
