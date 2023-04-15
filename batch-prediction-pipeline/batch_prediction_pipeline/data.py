from datetime import datetime
from typing import Tuple

import pandas as pd

from hsfs.feature_store import FeatureStore


def load_data_from_feature_store(
    fs: FeatureStore,
    feature_view_version: int,
    start_datetime: datetime,
    end_datetime: datetime,
    target: str = "energy_consumption",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data for a given time range from the feature store.

    Args:
        fs: Feature store.
        feature_view_version: Feature view version.
        start_datetime: Start datetime.
        end_datetime: End datetime.
        target: Name of the target feature.

    Returns:
        Tuple of exogenous variables and the time series to be forecasted.
    """

    feature_view = fs.get_feature_view(
        name="energy_consumption_denmark_view", version=feature_view_version
    )
    data = feature_view.get_batch_data(start_time=start_datetime, end_time=end_datetime)

    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    return X, y
