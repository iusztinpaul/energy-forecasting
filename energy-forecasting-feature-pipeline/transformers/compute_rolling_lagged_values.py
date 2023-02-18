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
    
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    n_lagged_rolling_days = kwargs.get("n_lagged_rolling_days", 1 / 24)
    n_lagged_rolling_hours = int(n_lagged_rolling_days * 24)

    rolling_columns = [c for c in df.columns if c.startswith("energy_consumption_rolling")]
    for rolling_column in rolling_columns:
        for n_lagged_hour in range(1, n_lagged_rolling_hours + 1, n_lagged_rolling_hours):
            lagged_df = df[["area", "consumer_type", "datetime_utc", rolling_column]].copy()
            lagged_df["datetime_utc"] = lagged_df["datetime_utc"] + pd.DateOffset(hours=n_lagged_hour)

            lagged_data_column_name = f"{rolling_column}_lagged_hours_{n_lagged_hour}"
            lagged_df = lagged_df.rename(columns={
                rolling_column: lagged_data_column_name
            })

            df = df.merge(lagged_df, how="left", on=["area", "consumer_type", "datetime_utc"], copy=False)
            df[lagged_data_column_name] = df[lagged_data_column_name].astype(df[rolling_column].dtype)
            
            # Drop the rolling column that contains information from the target value.
            df = df.drop(columns=[rolling_column])

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
