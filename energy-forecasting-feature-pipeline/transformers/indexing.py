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

    df["HourUTC"] = pd.to_datetime(df["HourUTC"])
    df = df.set_index("HourUTC")
    df = df.drop(columns=["HourDK"])

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
