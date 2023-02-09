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
    
    n_future_days = kwargs.get("n_future_days", 3)
    n_future_hours = n_future_days * 24
    n_future_hours_step = kwargs.get("n_future_hours_step", 1)

    df["UTC Datetime"] = pd.to_datetime(df["UTCDatetime"])

    for n_future_hour in range(1, n_future_hours + 1, n_future_hours_step):
        lagged_df = df[["Area", "ConsumerType", "UTCDatetime", "EnergyConsumption"]].copy()
        lagged_df["UTCDatetime"] = lagged_df["UTCDatetime"] - pd.DateOffset(hours=n_future_hour)

        lagged_data_column_name = f"EnergyConsumptionFutureHours{n_future_hour}"
        lagged_df = lagged_df.rename(columns={
            "EnergyConsumption": lagged_data_column_name
        })

        df = df.merge(lagged_df, how="left", on=["Area", "ConsumerType", "UTCDatetime"], copy=False)
        df[lagged_data_column_name] = df[lagged_data_column_name].astype(df["EnergyConsumption"].dtype)

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
