import pandas as pd


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature columns
    """

    data = df.copy()

    # Drop irrelevant columns.
    data.drop(columns=["HourDK"], inplace=True)

    # Rename columns
    data.rename(
        columns={
            "HourUTC": "datetime_utc",
            "PriceArea": "area",
            "ConsumerType_DE35": "consumer_type",
            "TotalCon": "energy_consumption",
        },
        inplace=True,
    )

    return data


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """

    data = df.copy()

    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])
    data["area"] = data["area"].astype("string")
    data["consumer_type"] = data["consumer_type"].astype("int32")
    data["energy_consumption"] = data["energy_consumption"].astype("float64")

    return data


def encode_area_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform string categorical data to numerical categorical data.

    Args:
        df (DataFrame): Data frame from parent block.

    Returns:
        DataFrame: Transformed data frame
    """

    data = df.copy()

    area_mappings = {"DK": 0, "DK1": 1, "DK2": 2}

    data["area"] = data["area"].map(lambda string_area: area_mappings.get(string_area))
    data["area"] = data["area"].astype("int8")

    return data
