
import json
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sktime.forecasting.model_evaluation import evaluate as cv_evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.utils.plotting import plot_windows

import utils
import wandb

from data import load_dataset_from_feature_store
from models import build_model
from utils import init_wandb_run
from settings import CREDENTIALS, OUTPUT_DIR


logger = utils.get_logger(__name__)


# TODO: Inject sweep configs from YAML
# TODO: Use random or bayesian search + early stopping
# sweep_configs = {
#     "method": "grid",
#     "metric": {"name": "validation.MAPE", "goal": "minimize"},
#     "parameters": {
#         "forecaster__estimator__n_jobs": {"values": [-1]},
#         "forecaster__estimator__n_estimators": {"values": [1500, 1700, 2000]},
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
        "forecaster__estimator__n_estimators": {"values": [1500, 2000, 100]},
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


# TODO: Inject fh and validation_metric_key from config.
def main(fh: int = 24):
    y_train, y_test, X_train, X_test = load_dataset_from_feature_store()

    find_best_model(
        y_train, X_train, fh=fh
    )


def find_best_model(
    y_train: pd.DataFrame,
    X_train: pd.DataFrame,
    fh: int,
) -> dict:
    sweep_id = run_hyperparameter_optimization(y_train, X_train, fh=fh)

    api = wandb.Api()
    sweep = api.sweep(f"{CREDENTIALS['WANDB_ENTITY']}/{CREDENTIALS['WANDB_PROJECT']}/{sweep_id}")
    best_run = sweep.best_run()

    with init_wandb_run(
        name="best_experiment", job_type="hpo", group="train", run_id=best_run.id
    ) as run:
        # TODO: Try to initialize from scratch the run with the best config and NOT to override the last run.
        #   Otherwise it might override the wrong run.
        #   IDEA: Initialize the run with the id of the best_run --->
        #   !!! Now it takes the latest run, maybe the sweep.best_run() method is returning the wrong run.
        best_config_artifact = run.use_artifact("config:latest")
        best_config = best_config_artifact.metadata["config"]

        logger.info(f"Best run {best_run.name}")
        logger.info("Best run config:")
        logger.info(best_config)
        logger.info(
            f"Best run = {best_run.name} with results {best_config_artifact.metadata['results']}"
        )

        config_path = OUTPUT_DIR / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=4)

        artifact = wandb.Artifact(
            name=f"best_config",
            type="model",
            metadata=best_config_artifact.metadata,
        )
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)

        run.finish()

    return best_config


def run_hyperparameter_optimization(
    y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24
):
    sweep_id = wandb.sweep(sweep=sweep_configs, project=CREDENTIALS["WANDB_PROJECT"])

    wandb.agent(
        project=CREDENTIALS["WANDB_PROJECT"],
        sweep_id=sweep_id,
        function=partial(run_sweep, y_train=y_train, X_train=X_train, fh=fh),
    )

    return sweep_id


def run_sweep(y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24):
    with init_wandb_run(name="experiment", job_type="hpo", group="train", add_timestamp_to_name=True) as run:
        run.use_artifact("split_train:latest")

        config = wandb.config
        config = dict(config)
        model = build_model(config)

        model, results = train_model_cv(model, y_train, X_train, fh=fh)
        wandb.log(results)

        metadata = {
            "experiment": {
                "name": run.name,
            },
            "results": results,
            "config": config,
        }
        artifact = wandb.Artifact(
            name=f"config",
            type="model",
            metadata=metadata,
        )
        run.log_artifact(artifact)

        run.finish()


def train_model_cv(
    model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int = 24, k: int = 3
):
    data_length = len(y_train.index.get_level_values(-1).unique())
    assert data_length >= fh * 10, "Not enough data to perform a 3 fold CV."

    cv_step_length = data_length // k
    initial_window = max(fh * 3, cv_step_length - fh)
    cv = ExpandingWindowSplitter(
        step_length=cv_step_length, fh=np.arange(fh) + 1, initial_window=initial_window
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


def render_cv_scheme(cv, y_train: pd.DataFrame) -> str:
    random_time_series = (
        y_train.groupby(level=[0, 1])
        .get_group((1, 111))
        .reset_index(level=[0, 1], drop=True)
    )
    plot_windows(cv, random_time_series)

    save_path = str(OUTPUT_DIR / "cv_scheme.png")
    plt.savefig(save_path)
    wandb.log({"cv_scheme": wandb.Image(save_path)})

    return save_path


if __name__ == "__main__":
    main()
