import numpy as np
import pandas as pd


from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Transform string categorical data to numerical categorical data.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """
    
    area_mappings = {
        "DK": 0,
        "DK1": 1,
        "DK2": 2
    }
    df["area"] = df["area"].map(lambda string_area: area_mappings.get(string_area))
    df["area"] = df["area"].astype("int8")
    df["consumer_type"] = df["consumer_type"].astype("int32")

    return df


@test
def test_output(df, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert df is not None, 'The output is undefined'

    assert df["area"].isna().any() is np.bool_(False), "Found unsupported area values in the DataFrame."
