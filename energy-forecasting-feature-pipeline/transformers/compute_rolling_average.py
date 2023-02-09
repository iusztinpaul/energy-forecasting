import pandas as pd

from functools import partial
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

    days_rolling_average = kwargs.get("days_rolling_average", 1)
    # Convert days to hours
    hours_rolling_average = days_rolling_average * 24 

    df[f"EnergyConsumptionRollingAverageDays{days_rolling_average}"] = df.\
        groupby(["Area", "ConsumerType"])["EnergyConsumption"].\
        transform(lambda x: x.rolling(hours_rolling_average, min_periods=hours_rolling_average).mean())
    
    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'
