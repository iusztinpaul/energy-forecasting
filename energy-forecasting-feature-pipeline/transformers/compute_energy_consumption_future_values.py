import pandas as pd

from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Function that computes the target columns in the 'EnergyConsumptionFutureHours{hour}' format.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    n_future_days = kwargs.get("n_future_days", 0)
    n_future_hours = n_future_days * 24
    n_future_hours_step = kwargs.get("n_future_hours_step", 1)

    for n_future_hour in range(1, n_future_hours + 1, n_future_hours_step):
        lagged_df = df[["area", "consumer_type", "datetime_utc", "energy_consumption"]].copy()
        lagged_df["datetime_utc"] = lagged_df["datetime_utc"] - pd.DateOffset(hours=n_future_hour)

        lagged_data_column_name = f"energy_consumption_future_hours_{n_future_hour}"
        lagged_df = lagged_df.rename(columns={
            "energy_consumption": lagged_data_column_name
        })

        df = df.merge(lagged_df, how="left", on=["area", "consumer_type", "datetime_utc"], copy=False)
        df[lagged_data_column_name] = df[lagged_data_column_name].astype(df["energy_consumption"].dtype)

    # Rename the original column for consistency.
    df = df.rename(columns={"energy_consumption": "energy_consumption_future_hours_0"})

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
