from typing import Tuple

import pandas as pd
from category_encoders import hashing, one_hot


def split_data(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Template for splitting data into train and test sets.

    Args:
        df: Dataframe containing the data.

    Returns: Tuple of train and test dataframes.
    """

    test_split_size_days = kwargs.get("test_split_size_days", 1)
    validation_split_size_days = kwargs.get("validation_split_size_days", 1)

    max_datetime = df["datetime_utc"].max()
    min_test_split_datetime = max_datetime - pd.DateOffset(days=test_split_size_days)
    min_validation_split_datetime = max_datetime - pd.DateOffset(days=test_split_size_days + validation_split_size_days)

    train_mask = df["datetime_utc"] < min_validation_split_datetime
    validation_mask = df["datetime_utc"].between(
        min_validation_split_datetime,
        min_test_split_datetime,
        inclusive="left"
    )
    test_mask = df["datetime_utc"] >= min_test_split_datetime

    train_df = df[train_mask]
    validation_df = df[validation_mask]
    test_df = df[test_mask]

    return train_df, validation_df, test_df


def encode_categorical(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical variables

    Args:
        train_df (DataFrame): Data frame containing the training split
        validation_df (DataFrame): Data frame containing the validation split
        test_df (DataFrame): Data frame containing the testing split

    Returns:
        DataFrame: Transformed data frames
    """

    consumer_type_hashing_encoder = hashing.HashingEncoder(return_df=True, cols=["consumer_type"])
    train_df = consumer_type_hashing_encoder.fit_transform(train_df)
    validation_df = consumer_type_hashing_encoder.transform(validation_df)
    test_df = consumer_type_hashing_encoder.transform(test_df)

    area_one_hot_encoder = one_hot.OneHotEncoder(return_df=True, cols=["area"], handle_unknown="error")
    train_df = area_one_hot_encoder.fit_transform(train_df)
    validation_df = area_one_hot_encoder.fit_transform(validation_df)
    test_df = area_one_hot_encoder.transform(test_df)

    return train_df, validation_df, test_df
