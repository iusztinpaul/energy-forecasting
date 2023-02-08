import pandas as pd

from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    
    n_past_days = kwargs.get("n_past_days", 3)
    n_past_hours = n_past_days * 24
    n_past_hours_step = kwargs.get("n_past_hours_step", 3)

    df["UTC Datetime"] = pd.to_datetime(df["UTC Datetime"])

    for n_past_hour in range(1, n_past_hours + 1, n_past_hours_step):
        lagged_df = df[["Area", "Consumer Type", "UTC Datetime", "Energy Consumption"]].copy()
        lagged_df["UTC Datetime"] = lagged_df["UTC Datetime"] + pd.DateOffset(hours=n_past_hour)
        lagged_df = lagged_df.rename(columns={
            "Energy Consumption": f"Energy Consumption {n_past_hour}"
        })

        df = df.merge(lagged_df, how="left", on=["Area", "Consumer Type", "UTC Datetime"])

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
