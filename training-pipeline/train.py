import json
from collections import OrderedDict
from pathlib import Path
from typing import OrderedDict as OrderedDictType, Optional

import hopsworks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sktime.performance_metrics.forecasting import (
    mean_squared_percentage_error,
    mean_absolute_percentage_error,
)
from sktime.utils.plotting import plot_series


import utils

from settings import CREDENTIALS, OUTPUT_DIR
from data import load_dataset_from_feature_store
from models import build_model, build_baseline_model


logger = utils.get_logger(__name__)


# TODO: Inject fh from config.
def main(fh: int = 24):
    y_train, y_test, X_train, X_test = load_dataset_from_feature_store()

    with utils.init_wandb_run(
        name="best_model", job_type="train_best_model", group="train", reinit=True, add_timestamp_to_name=True
    ) as run:
        run.use_artifact("split_train:latest")
        run.use_artifact("split_test:latest")
        # Load the best config from sweep.
        best_config_artifact = run.use_artifact(
            "best_config:latest",
            type="model",
        )
        download_dir = best_config_artifact.download()
        config_path = Path(download_dir) / "best_config.json"
        with open(config_path) as f:
            config = json.load(f)
        # Log the config to the experiment.
        run.config.update(config)

        # Baseline model
        baseline_forecaster = build_baseline_model()
        baseline_forecaster = train_model(baseline_forecaster, y_train, X_train, fh=fh)
        y_pred_baseline, metrics_baseline = evaluate(baseline_forecaster, y_test, X_test)
        for k, v in metrics_baseline.items():
            logger.info(f"Baseline test {k.upper()}: {v}")
        wandb.log({
            "test": {
                "baseline": metrics_baseline
            }
        })

        # Build & train best model.
        best_model = build_model(config)
        best_forecaster = train_model(best_model, y_train, X_train, fh=fh)

        # # Evaluate best model
        y_pred, metrics = evaluate(best_forecaster, y_test, X_test)
        for k, v in metrics.items():
            logger.info(f"Model test {k}: {v}")
        wandb.log({
            "test": {
                "model": metrics
            }
        })

        # Render best model on the test set.
        results = OrderedDict({"y_train": y_train, "y_test": y_test, "y_pred": y_pred})
        render(results, prefix="images_test")

        # Update best model with the test set.
        best_forecaster = best_forecaster.update(y_test, X=X_test)
        # TODO: Make the forecast function independent from X_test.
        y_forecast = forecast(best_forecaster, X_test)
        results = OrderedDict(
            {
                "y_train": y_train,
                "y_test": y_test,
                "y_forecast": y_forecast,
            }
        )
        # Render best model future forecasts.
        render(results, prefix="images_forecast")

        # Save best model.
        save_model_path = OUTPUT_DIR / "best_model.pkl"
        utils.save_model(best_forecaster, save_model_path)
        metadata = {"results": {"test": metrics}}
        artifact = wandb.Artifact(name="best_model", type="model", metadata=metadata)
        artifact.add_file(str(save_model_path))
        run.log_artifact(artifact)

        run.finish()

    attach_best_model_to_feature_store()


def train_model(model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24):
    fh = np.arange(fh) + 1

    model.fit(y_train, X=X_train, fh=fh)

    return model


def evaluate(forecaster, y_test: pd.DataFrame, X_test: pd.DataFrame):
    y_pred = forecaster.predict(X=X_test)

    rmspe = mean_squared_percentage_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)

    return y_pred, {"rmspe": rmspe, "mape": mape}


def render(
    timeseries: OrderedDictType[str, pd.DataFrame], prefix: Optional[str] = None
):
    grouped_timeseries = OrderedDict()
    for split, df in timeseries.items():
        df = df.reset_index(level=[0, 1])
        groups = df.groupby(["area", "consumer_type"])
        for group_name, split_group_values in groups:
            group_values = grouped_timeseries.get(group_name, {})

            grouped_timeseries[group_name] = {
                f"{split}": split_group_values["energy_consumption"],
                **group_values,
            }

    output_dir = OUTPUT_DIR / prefix if prefix else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, group_values_dict in grouped_timeseries.items():
        fig, ax = plot_series(
            *group_values_dict.values(), labels=group_values_dict.keys()
        )
        fig.suptitle(f"Area: {group_name[0]} - Consumer type: {group_name[1]}")

        # save matplotlib image
        image_save_path = str(output_dir / f"{group_name[0]}_{group_name[1]}.png")
        plt.savefig(image_save_path)
        plt.close(fig)

        if prefix:
            wandb.log({prefix: wandb.Image(image_save_path)})
        else:
            wandb.log(wandb.Image(image_save_path))


def forecast(forecaster, X_test, fh: int = 24):
    # TODO: Make this function independent from X_test.
    X_forecast = X_test.copy()
    X_forecast.index.set_levels(
        X_forecast.index.levels[-1] + fh, level=-1, inplace=True
    )

    y_forecast = forecaster.predict(X=X_forecast)

    return y_forecast


def attach_best_model_to_feature_store():
    project = hopsworks.login(api_key_value=CREDENTIALS["FS_API_KEY"], project="energy_consumption")
    fs = project.get_feature_store()

    feature_views = fs.get_feature_views("energy_consumption_denmark_view")
    feature_view = feature_views[-1]

    # TODO: The model is in hopsworks model registry. Do I still need all this logic?
    model_version = 0
    training_dataset_version = 2
    fs_tag = {
        "name": "best_model",
        "version": f"v{model_version}",
        "type": "model",
        "url": f"https://wandb.ai/{CREDENTIALS['WANDB_ENTITY']}/{CREDENTIALS['WANDB_PROJECT']}/artifacts/model/best_model/v{model_version}/overview",
        "artifact_name": f"{CREDENTIALS['WANDB_ENTITY']}/{CREDENTIALS['WANDB_PROJECT']}/best_model:latest",
    }
    feature_view.add_tag(name="wandb", value=fs_tag)
    feature_view.add_training_dataset_tag(
        training_dataset_version=1, name="wandb", value=fs_tag
    )

    feature_store_metadata = {
        "feature_view": feature_view.name,
        "feature_view_version": feature_view.version,
        "training_dataset_version": training_dataset_version,
    }

    api = wandb.Api()
    best_model_artifact = api.artifact(f"{CREDENTIALS['WANDB_ENTITY']}/{CREDENTIALS['WANDB_PROJECT']}/best_model:latest")
    best_model_artifact.metadata = {**best_model_artifact.metadata, "feature_store": feature_store_metadata}
    best_model_artifact.save()

    # TODO: Add model schema & input example as docs.
    best_model_dir = best_model_artifact.download()
    best_model_path = Path(best_model_dir) / "best_model.pkl"
    best_model_metrics = best_model_artifact.metadata["results"]["test"]

    mr = project.get_model_registry()
    py_model = mr.python.create_model("best_model", metrics=best_model_metrics)
    py_model.save(best_model_path)


if __name__ == "__main__":
    main()
