import hopsworks
import pandas as pd

import wandb
from utils import init_wandb_run
from settings import CREDENTIALS


def load_dataset_from_feature_store(
    feature_view_version: int = 4, training_dataset_version: int = 1
):
    project = hopsworks.login(api_key_value=CREDENTIALS["FS_API_KEY"], project="energy_consumption")
    fs = project.get_feature_store()

    with init_wandb_run(
        name="load_training_data", job_type="load_feature_view", group="dataset"
    ) as run:
        # TODO: Get latest feature view version.
        feature_view = fs.get_feature_view(
            name="energy_consumption_denmark_view", version=feature_view_version
        )
        # TODO: Get the latest training dataset.
        # TODO: Handle hopsworks versions overall.
        data, _ = feature_view.get_training_data(
            training_dataset_version=training_dataset_version
        )

        fv_metadata = feature_view.to_dict()
        fv_metadata["query"] = fv_metadata["query"].to_string()
        fv_metadata["features"] = [f.name for f in fv_metadata["features"]]
        fv_metadata["link"] = feature_view._feature_view_engine._get_feature_view_url(
            feature_view
        )

        raw_data_at = wandb.Artifact(
            name="energy_consumption_denmark_feature_view",
            type="feature_view",
            metadata=fv_metadata,
        )
        run.log_artifact(raw_data_at)

        run.finish()

    with init_wandb_run(
        name="train_test_split", job_type="prepare_dataset", group="dataset"
    ) as run:
        run.use_artifact("energy_consumption_denmark_feature_view:latest")

        y_train, y_test, X_train, X_test = prepare_data(data)

        for split in ["train", "test"]:
            split_X = locals()[f"X_{split}"]
            split_y = locals()[f"y_{split}"]

            split_metadata = {
                "timespan": [split_X.index.min(), split_X.index.max()],
                "y_features": split_y.columns.tolist(),
                "X_features": split_X.columns.tolist(),
            }
            artifact = wandb.Artifact(
                name=f"split_{split}",
                type="split",
                metadata=split_metadata,
            )
            run.log_artifact(artifact)

        run.finish()

    return y_train, y_test, X_train, X_test


def prepare_data(data: pd.DataFrame, fh: int = 24):
    # TODO: Move these transformation to the FS. They are repeated in the batch prediction pipeline.

    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=["energy_consumption"])
    X["area_exog"] = X.index.get_level_values(0)
    X["consumer_type_exog"] = X.index.get_level_values(1)
    # Prepare the time series to be forecasted.
    y = data[["energy_consumption"]]

    y_train, y_test, X_train, X_test = create_train_test_split(y, X, fh=fh)

    return y_train, y_test, X_train, X_test


def create_train_test_split(y: pd.DataFrame, X: pd.DataFrame, fh: int):
    max_datetime = y.index.get_level_values(-1).max()
    min_datetime = max_datetime - fh + 1

    # TODO: Double check this mask.
    test_mask = y.index.get_level_values(-1) >= min_datetime
    train_mask = ~test_mask

    y_train = y.loc[train_mask]
    X_train = X.loc[train_mask]

    y_test = y.loc[test_mask]
    X_test = X.loc[test_mask]

    return y_train, y_test, X_train, X_test
