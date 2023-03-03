import json
import logging
import os
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import OrderedDict as OrderedDictType, Optional

import hopsworks
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from category_encoders import hashing
from dotenv import load_dotenv
from sktime.forecasting.compose import make_reduction, ForecastingPipeline
from sktime.forecasting.model_evaluation import evaluate as cv_evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    mean_squared_percentage_error,
    mean_absolute_percentage_error,
    MeanAbsolutePercentageError,
)
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.utils.plotting import plot_series, plot_windows

import utils

warnings.filterwarnings(action="ignore", category=FutureWarning, module="sktime")
matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="../.env.default")
load_dotenv(dotenv_path="../.env", override=True)

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
FS_API_KEY = os.getenv("FS_API_KEY")

# TODO: Change output dir with a tmp dir.
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO: Inject sweep configs from YAML
# TODO: Use random or bayesian search + early stopping
# sweep_configs = {
#     "method": "grid",
#     "metric": {"name": "validation.MAPE", "goal": "minimize"},
#     "parameters": {
#         "forecaster__estimator__n_jobs": {"values": [-1]},
#         "forecaster__estimator__n_estimators": {"values": [1000, 1200]},
#         "forecaster__estimator__learning_rate": {"values": [0.1, 0.15]},
#         "forecaster__estimator__max_depth": {"values": [-1, 4]},
#         "forecaster__estimator__reg_lambda": {"values": [0.0, 0.01]},
#         "daily_season__manual_selection": {"values": [["day_of_week", "hour_of_day"]]},
#         "forecaster_transformers__window_summarizer__lag_feature__lag": {
#             "values": [list(range(1, 25)), list(range(1, 49)), list(range(1, 73))]
#         },
#         "forecaster_transformers__window_summarizer__lag_feature__mean": {
#             "values": [[[1, 24], [1, 48]], [[1, 24], [1, 48], [1, 72]]]
#         },
#         "forecaster_transformers__window_summarizer__lag_feature__std": {
#             "values": [[[1, 24], [1, 48]], [[1, 24], [1, 48], [1, 72]]]
#         },
#         "forecaster_transformers__window_summarizer__n_jobs": {"values": [-1]},
#     },
# }

sweep_configs = {
    "method": "grid",
    "metric": {"name": "validation.MAPE", "goal": "minimize"},
    "parameters": {
        "forecaster__estimator__n_jobs": {"values": [-1]},
        "forecaster__estimator__n_estimators": {"values": [1200]},
        "forecaster__estimator__learning_rate": {"values": [0.15]},
        "forecaster__estimator__max_depth": {"values": [-4]},
        "forecaster__estimator__reg_lambda": {"values": [0.01]},
        "daily_season__manual_selection": {"values": [["day_of_week", "hour_of_day"]]},
        "forecaster_transformers__window_summarizer__lag_feature__lag": {
            "values": [list(range(1, 73))]
        },
        "forecaster_transformers__window_summarizer__lag_feature__mean": {
            "values": [[[1, 24], [1, 48], [1, 72]]]
        },
        "forecaster_transformers__window_summarizer__lag_feature__std": {
            "values": [[[1, 24], [1, 48]]]
        },
        "forecaster_transformers__window_summarizer__n_jobs": {"values": [1]},
    },
}


def main(fh: int = 24, validation_metric_key: str = "MAPE"):
    y_train, y_test, X_train, X_test = get_dataset_hopsworks()

    baseline_forecaster = build_baseline_model()
    baseline_forecaster = train_model(baseline_forecaster, y_train, X_train)
    y_pred_baseline, metrics_baseline = evaluate(baseline_forecaster, y_test, X_test)
    for k, v in metrics_baseline.items():
        logger.info(f"Baseline test {k.upper()}: {v}")

    find_best_model(
        y_train, X_train, fh=fh, validation_metric_key=validation_metric_key
    )

    best_forecaster = build_model_from_artifact()
    with wandb.init(
        name="train_best_model", job_type="train_best_model", group="model"
    ) as run:
        best_forecaster = train_model(best_forecaster, y_train, X_train)
        save_model_path = OUTPUT_DIR / "best_model.pkl"

        y_pred, metrics = evaluate(best_forecaster, y_test, X_test)
        for k, v in metrics.items():
            logger.info(f"Model test {k.upper()}: {v}")

        results = OrderedDict({"y_train": y_train, "y_test": y_test, "y_pred": y_pred})
        render(results, prefix="images_test")

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
        render(results, prefix="images_forecast")

        utils.save_model(best_forecaster, save_model_path)
        metadata = {"results": {"test": metrics}}
        artifact = wandb.Artifact(name="best_model", type="model", metadata=metadata)
        artifact.add_file(str(save_model_path))
        run.log_artifact(artifact)

    attach_best_model_to_feature_store()


def get_dataset_hopsworks():
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()

    feature_views = fs.get_feature_views("energy_consumption_denmark_view")
    feature_view = feature_views[-1]
    # TODO: Get the latest training dataset.
    # TODO: Handle hopsworks versions overall.
    X, y = feature_view.get_training_data(training_dataset_version=1)

    with init_wandb_run(
        name="feature_view", job_type="load_dataset", group="dataset"
    ) as run:
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

    with init_wandb_run(
        name="train_test_split", job_type="prepare_dataset", group="dataset"
    ) as run:
        y_train, y_test, X_train, X_test = prepare_data(X, y)

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

    return y_train, y_test, X_train, X_test


def prepare_data(X: pd.DataFrame, y: pd.DataFrame, fh: int = 24):
    # TODO: Move these transformation to the FS. They are repeated in the batch prediction pipeline.

    # Set the index as is required by sktime.
    data = pd.concat([X, y], axis=1)
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


def find_best_model(
    y_train: pd.DataFrame,
    X_train: pd.DataFrame,
    fh: int = 24,
    validation_metric_key: str = "MAPE",
) -> dict:
    sweep_id = run_hyperparameter_optimization(y_train, X_train, fh=fh)

    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
    best_run = sweep.best_run()
    config = dict(best_run.config)

    with init_wandb_run(
        name="config", job_type="find_best_model", group="model"
    ) as run:
        config_path = OUTPUT_DIR / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        metric_value = best_run.summary.get("validation", {}).get(
            validation_metric_key, 0
        )
        metadata = {
            "experiment": {
                "name": best_run.name,
            },
            "results": {
                "validation": {
                    validation_metric_key: metric_value,
                }
            },
        }
        artifact = wandb.Artifact(
            name=f"best_model_config",
            type="model",
            metadata=metadata,
        )
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)

    logger.info(f"Best run {best_run.name}")
    logger.info("Best run config:")
    logger.info(config)

    logger.info(
        f"Best run = {best_run.name} with validation {validation_metric_key} = {metric_value}"
    )

    return config


def run_hyperparameter_optimization(
    y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24
):
    sweep_id = wandb.sweep(sweep=sweep_configs, project=WANDB_PROJECT)

    wandb.agent(
        project=WANDB_PROJECT,
        sweep_id=sweep_id,
        function=partial(run_sweep, y_train=y_train, X_train=X_train, fh=fh),
    )

    return sweep_id


def run_sweep(y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24):
    with init_wandb_run(name="experiment", job_type="hpo", add_timestamp_to_name=True):
        config = wandb.config

        model, results = build_and_train_model_cv(dict(config), y_train, X_train, fh=fh)

        wandb.log(results)


def build_and_train_model_cv(
    config: dict, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24
):
    model = build_model(config)
    model, results = train_model_cv(model, y_train, X_train, fh=fh)

    return model, results


def train_model_cv(model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24):
    # TODO: Find a smarter way to compute a 3 fold CV.
    cv = ExpandingWindowSplitter(
        step_length=fh * 9, fh=np.arange(fh) + 1, initial_window=fh * 4
    )
    render_cv_scheme(cv, y_train)

    results = cv_evaluate(
        forecaster=model,
        y=y_train,
        X=X_train,
        cv=cv,
        strategy="refit",
        scoring=MeanAbsolutePercentageError(symmetric=False),
        error_score="raise",
        return_data=False,
    )

    results = results.rename(
        columns={
            "test_MeanAbsolutePercentageError": "MAPE",
            "fit_time": "fit_time",
            "pred_time": "prediction_time",
        }
    )
    mean_results = results[["MAPE", "fit_time", "prediction_time"]].mean(axis=0)
    mean_results = mean_results.to_dict()
    results = {"validation": mean_results}

    logger.info(f"Validation MAPE: {results['validation']['MAPE']:.2f}")
    logger.info(f"Mean fit time: {results['validation']['fit_time']:.2f} s")
    logger.info(f"Mean predict time: {results['validation']['prediction_time']:.2f} s")

    return model, results


def train_model(model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24):
    fh = np.arange(fh) + 1

    model.fit(y_train, X=X_train, fh=fh)

    return model


def render_cv_scheme(cv, y_train: pd.DataFrame):
    random_time_series = (
        y_train.groupby(level=[0, 1])
        .get_group((1, 111))
        .reset_index(level=[0, 1], drop=True)
    )
    plot_windows(cv, random_time_series)

    plt.savefig(OUTPUT_DIR / f"cv_scheme.png")

    # TODO: Save to wandb


def build_baseline_model():
    forecaster = NaiveForecaster(sp=24)

    return forecaster


def build_model_from_artifact():
    with init_wandb_run(name="config", job_type="best_model", group="model") as run:
        artifact = run.use_artifact(
            "best_model_config:latest",
            type="model",
        )
        download_dir = artifact.download()
        config_path = Path(download_dir) / "best_config.json"
        with open(config_path) as f:
            config = json.load(f)

    model = build_model(config)

    return model


def build_model(config: dict):
    lag = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__lag",
        list(range(1, 72 + 1)),
    )
    mean = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__mean",
        [[1, 24], [1, 48], [1, 72]],
    )
    std = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__std",
        [[1, 24], [1, 48], [1, 72]],
    )
    n_jobs = config.pop("forecaster_transformers__window_summarizer__n_jobs", 1)
    window_summarizer = WindowSummarizer(
        **{"lag_feature": {"lag": lag, "mean": mean, "std": std}},
        n_jobs=n_jobs,
    )

    regressor = lgb.LGBMRegressor()
    forecaster = make_reduction(
        regressor,
        transformers=[window_summarizer],
        strategy="recursive",
        pooling="global",
        window_length=None,
    )

    pipe = ForecastingPipeline(
        steps=[
            (
                "daily_season",
                DateTimeFeatures(
                    manual_selection=["day_of_week", "hour_of_day"],
                    keep_original_columns=True,
                ),
                (
                    "encode_categorical",
                    TabularToSeriesAdaptor(
                        hashing.HashingEncoder(
                            return_df=True,
                            cols=[
                                "area",
                                "consumer_type",
                                "day_of_week",
                                "hour_of_day",
                            ],
                        )
                    ),
                ),
            ),
            ("forecaster", forecaster),
        ]
    )
    pipe = pipe.set_params(**config)

    return pipe


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
        plt.savefig(output_dir / f"{group_name[0]}_{group_name[1]}.png")
        plt.close(fig)

        # TODO: Save to wandb.


def forecast(forecaster, X_test, fh: int = 24):
    X_forecast = X_test.copy()
    X_forecast.index.set_levels(
        X_forecast.index.levels[-1] + fh, level=-1, inplace=True
    )

    y_forecast = forecaster.predict(X=X_forecast)

    return y_forecast


def init_wandb_run(
    name: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    add_timestamp_to_name: bool = False,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
    **kwargs,
):
    if add_timestamp_to_name:
        name = f"{name}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        **kwargs,
    )

    return run


def attach_best_model_to_feature_store():
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()

    feature_views = fs.get_feature_views("energy_consumption_denmark_view")
    feature_view = feature_views[-1]

    model_version = 0
    training_dataset_version = 2
    fs_tag = {
        "name": "best_model",
        "version": f"v{model_version}",
        "type": "model",
        "url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/artifacts/model/best_model/v{model_version}/overview",
        "artifact_name": f"{WANDB_ENTITY}/{WANDB_PROJECT}/best_model:latest",
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
    artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/best_model:latest")
    artifact.metadata = {**artifact.metadata, "feature_store": feature_store_metadata}
    artifact.save()

    artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/best_model_config:latest")
    artifact.metadata = {**artifact.metadata, "feature_store": feature_store_metadata}
    artifact.save()

    # TODO: Also save model to Hopsworks model registry.


if __name__ == "__main__":
    main()
