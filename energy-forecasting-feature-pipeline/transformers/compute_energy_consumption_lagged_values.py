import pandas as pd

from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Compute lagged values for the energy consumption time series.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    
    n_lagged_days = kwargs.get("n_lagged_days", 3)
    n_lagged_hours = n_lagged_days * 24
    n_lagged_hours_step = kwargs.get("n_lagged_hours_step", 3)

    for n_lagged_hour in range(1, n_lagged_hours + 1, n_lagged_hours_step):
        lagged_df = df[["area", "consumer_type", "datetime_utc", "energy_consumption"]].copy()
        lagged_df["datetime_utc"] = lagged_df["datetime_utc"] + pd.DateOffset(hours=n_lagged_hour)

        lagged_data_column_name = f"energy_consumption_lagged_hours_{n_lagged_hour}"
        lagged_df = lagged_df.rename(columns={
            "energy_consumption": lagged_data_column_name
        })

        df = df.merge(lagged_df, how="left", on=["area", "consumer_type", "datetime_utc"], copy=False)
        df[lagged_data_column_name] = df[lagged_data_column_name].astype(df["energy_consumption"].dtype)

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
