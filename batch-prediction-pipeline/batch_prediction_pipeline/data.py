from datetime import datetime

import pandas as pd

from hsfs.feature_store import FeatureStore



def load_data_from_feature_store(
    fs: FeatureStore,
    feature_view_version: int,
    start_datetime: datetime,
    end_datetime: datetime,
    target: str = "energy_consumption",
):
    feature_view = fs.get_feature_view(
        name="energy_consumption_denmark_view", version=feature_view_version
    )

    data = feature_view.get_batch_data(start_time=start_datetime, end_time=end_datetime)

    # TODO: Can I move the index preparation to the model pipeline?
    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    return X, y
