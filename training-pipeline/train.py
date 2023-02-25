import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import OrderedDict as OrderedDictType, Optional

import hopsworks
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_squared_percentage_error
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.utils.plotting import plot_series

import preprocess
import utils

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

WANDB_ENTITY = "p-b-iusztin"
WANDB_PROJECT = "energy_consumption"
FS_API_KEY = os.getenv("FS_API_KEY")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sweep_configs = {
    "method": "grid",
    "metric": {"name": "validation.rmse", "goal": "minimize"},
    "parameters": {
        "n_estimators": {"values": [400, 800]},
        "max_depth": {"values": [3, 5]},
        "learning_rate": {"values": [0.05, 0.1]},
        "bagging_fraction": {"values": [0.8, 1.0]},
        "feature_fraction": {"values": [0.8, 1.0]},
        "lambda_l2": {"values": [0.0, 0.01]},
    },
}
# sweep_configs = {
#     "method": "grid",
#     "metric": {"name": "validation.rmse", "goal": "minimize"},
#     "parameters": {
#         "n_estimators": {"values": [1000]},
#         "max_depth": {"values": [3]},
#         "learning_rate": {"values": [0.05]},
#         "bagging_fraction": {"values": [0.8]},
#         "feature_fraction": {"values": [0.8]},
#         "lambda_l2": {"values": [0.0]},
#     },
# }
# sweep_id = wandb.sweep(sweep=sweep_configs, project="energy_consumption")


def main():
    y_train, y_test, X_train, X_test = get_dataset_hopsworks()

    # baseline_forecaster = train_baseline(y_train)
    # y_pred_baseline, metrics_baseline = evaluate(baseline_forecaster, y_test)
    # logger.info(f"Baseline RMPE: {metrics_baseline}")

    forecaster = train_sweep(y_train)
    y_pred, metrics = evaluate(forecaster, y_test)
    logger.info(f"Model RMPE: {metrics}")

    results = OrderedDict({
        "y_train": y_train,
        "y_test": y_test,
        # "y_pred_baseline": y_pred_baseline,
        "y_pred": y_pred
    })
    render(results, prefix="images_test")

    forecaster = forecaster.update(y_test)

    y_forecast = forecast(forecaster)
    results = OrderedDict({
        "y_train": y_train,
        "y_test": y_test,
        "y_forecast": y_forecast,
    })
    render(results, prefix="images_forecast")


def get_dataset_hopsworks():
    project = hopsworks.login(api_key_value=FS_API_KEY, project="energy_consumption")
    fs = project.get_feature_store()

    # Get the train, validation and test splits.
    feature_views = fs.get_feature_views("energy_consumption_denmark_view")
    feature_view = feature_views[-1]
    X, y = feature_view.get_training_data(
        training_dataset_version=1
    )

    # TODO: Call close?

    data = pd.concat([X, y], axis=1)
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    X = data.drop(columns=["energy_consumption"])
    y = data[["energy_consumption"]]

    y_train, y_test, X_train, X_test = prepare_data(X, y)

    return y_train, y_test, X_train, X_test


def prepare_data(X: pd.DataFrame, y: pd.DataFrame, periods: int = 24):
    # Split data
    y_train, y_test, X_train, X_test = create_train_test_split(y, X, periods=periods)

    return y_train, y_test, X_train, X_test


def create_train_test_split(y: pd.DataFrame, X: pd.DataFrame, periods: int):
    max_datetime = y.index.get_level_values(-1).max()
    min_datetime = max_datetime - periods + 1

    # TODO: Double check this mask.
    test_mask = y.index.get_level_values(-1) >= min_datetime
    train_mask = ~test_mask

    y_train = y.loc[train_mask]
    X_train = X.loc[train_mask]

    y_test = y.loc[test_mask]
    X_test = X.loc[test_mask]

    return y_train, y_test, X_train, X_test


def train_baseline(y_train: pd.DataFrame, periods: int = 24):
    fh = np.arange(periods) + 1

    forecaster = NaiveForecaster(strategy="last", sp=fh)
    forecaster.fit(y_train, fh=fh)

    return forecaster


def train_sweep(y_train: pd.DataFrame, periods: int = 24):
    # forecaster = ForecastingPipeline(
    #     [("hampel", HampelFilter()),
    #      ("log", LogTransformer()),
    #      ("forecaster", ThetaForecaster(sp=24))])

    fh = np.arange(periods) + 1

    kwargs = {
        "lag_feature": {
            "lag": [1, 2, 3, 4, 8, 12, 16, 24, 48, 72],
            "mean": [[1, 24], [1, 48], [1, 72]],
            "std": [[1, 24], [1, 48], [1, 72]],
        }
    }

    regressor = lgb.LGBMRegressor()
    forecaster = make_reduction(
        regressor,
        transformers=[WindowSummarizer(**kwargs, n_jobs=1)],
        strategy="recursive",
        pooling="global",
        window_length=None
    )
    forecaster.fit(y_train, fh=fh)

    return forecaster


def evaluate(forecaster, y_test: pd.DataFrame):
    y_pred = forecaster.predict()

    rspe = mean_squared_percentage_error(y_test, y_pred, squared=False)

    return y_pred, rspe


def render(timeseries: OrderedDictType[str, pd.DataFrame], prefix: Optional[str] = None):

    grouped_timeseries = OrderedDict()
    for split, df in timeseries.items():
        df = df.reset_index(level=[0, 1])
        groups = df.groupby(["area", "consumer_type"])
        for group_name, split_group_values in groups:
            group_values = grouped_timeseries.get(group_name, {})

            grouped_timeseries[group_name] = {
                f"{split}": split_group_values["energy_consumption"],
                **group_values
            }

    output_dir = OUTPUT_DIR / prefix if prefix else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, group_values_dict in grouped_timeseries.items():
        fig, ax = plot_series(*group_values_dict.values(), labels=group_values_dict.keys())
        fig.suptitle(f"Area: {group_name[0]} - Consumer type: {group_name[1]}")

        # save matplotlib image
        plt.savefig(output_dir / f"{group_name[0]}_{group_name[1]}.png")
        plt.close(fig)

def forecast(forecaster):
    y_forecast = forecaster.predict()

    return y_forecast

# def get_dataset(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     # load data
#     df = utils.load_data_from_parquet(data_path)
#     # preprocess data
#     train_df, validation_df, test_df = preprocess.split_data(df)
#     train_df, validation_df, test_df = preprocess.encode_categorical(
#         train_df, validation_df, test_df
#     )
#
#     metadata = {
#         "features": list(df.columns),
#     }
#     with init_wandb_run(name="feature_view", job_type="upload_feature_view") as run:
#         raw_data_at = wandb.Artifact(
#             "energy_consumption_data",
#             type="feature_view",
#             metadata=metadata,
#         )
#         run.log_artifact(raw_data_at)
#
#     with init_wandb_run(name="train_validation_test_split", job_type="split") as run:
#         data_at = run.use_artifact("energy_consumption_data:latest")
#         data_dir = data_at.download()
#
#         artifacts = {}
#         for split in ["train", "validation", "test"]:
#             split_df = locals()[f"{split}_df"]
#             starting_datetime = split_df["datetime_utc"].min()
#             ending_datetime = split_df["datetime_utc"].max()
#             metadata = {
#                 "starting_datetime": starting_datetime,
#                 "ending_datetime": ending_datetime,
#             }
#
#             artifacts[split] = wandb.Artifact(
#                 f"{split}_split", type="split", metadata=metadata
#             )
#
#         for split, artifact in artifacts.items():
#             run.log_artifact(artifact)
#
#     return train_df, validation_df, test_df


# def train_sweep(
#     train_df: pd.DataFrame,
#     validation_df: pd.DataFrame,
#     target: str = "energy_consumption_future_hours_0",
# ) -> lgb.LGBMRegressor:
#     """
#     Function that is training a LGBM model.
#
#      Args:
#          train_df: Training data.
#          validation_df: Validation data.
#          target: Name of the target column.
#
#      Returns: Trained LightGBM model
#     """
#
#     with init_wandb_run(name="experiment", job_type="hpo") as run:
#         config = wandb.config
#         data_at = run.use_artifact(
#             f"{WANDB_ENTITY}/{WANDB_PROJECT}/test_split:latest", type="split"
#         )
#         data_at = run.use_artifact(
#             f"{WANDB_ENTITY}/{WANDB_PROJECT}/train_split:latest", type="split"
#         )
#
#         model = train_lgbm_regressor(df=train_df, target=target, config=config)
#
#         # evaluate model
#         rmse = evaluate_model(model, train_df, target=target)
#         logger.info(f"Train RMSE: {rmse:.2f}")
#         logger.info(
#             f"Train Mean Energy Consumption: {train_df['energy_consumption_future_hours_0'].mean():.2f}"
#         )
#         wandb.log({"train": {"rmse": rmse}})
#
#         rmse = evaluate_model(model, validation_df, target=target)
#         logger.info(f"Validation RMSE: {rmse:.2f}")
#         logger.info(
#             f"Validation Mean Energy Consumption: {validation_df['energy_consumption_future_hours_0'].mean():.2f}"
#         )
#         wandb.log({"validation": {"rmse": rmse}})
#
#     return model


def train_lgbm_regressor(
    df: pd.DataFrame, target: str, config: dict, **kwargs
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective=config.get("objective", "regression"),
        metric=config.get("metric", "rmse"),
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        bagging_fraction=config["bagging_fraction"],
        feature_fraction=config["feature_fraction"],
        lambda_l2=config["lambda_l2"],
        n_jobs=-1,
        random_state=42,
        **kwargs,
    )

    feature_columns = list(set(df.columns) - set([target, "datetime_utc"]))
    model.fit(X=df[feature_columns], y=df[target])

    return model


def evaluate_model(model, df: pd.DataFrame, target: str):
    """
    Template for evaluating a model.

    Args:
        model: Trained model.
        df: Dataframe containing the evaluation data.
        target: Name of the target column.

    Returns: RMSE
    """

    feature_columns = list(set(df.columns) - set([target, "datetime_utc"]))
    y_pred = model.predict(df[feature_columns])
    y_true = df[target]

    return mean_squared_error(y_true, y_pred, squared=False)


def init_wandb_run(
    name: str, project: str = WANDB_PROJECT, entity: str = WANDB_ENTITY, **kwargs
):
    name = f"{name}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(project=project, entity=entity, name=name, **kwargs)

    return run


def train_best_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_model_dir: str = "../models",
    target: str = "energy_consumption_future_hours_0",
) -> None:
    """
    Template for training a model.

    Args:
        train_df: Training data.
        validation_df: Validation data.
        test_df: Test data.
        save_model_dir: Directory where the models will be saved.
        target: Name of the target column.
    """

    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{sweep_id}")
    best_run = sweep.best_run()
    config = best_run.config
    config_quantile = config.copy()
    config_quantile["objective"] = "quantile"
    config_quantile["metric"] = "quantile"

    # train on everything for the final model
    save_model_dir = Path(save_model_dir)
    save_model_dir.mkdir(parents=True, exist_ok=True)

    with init_wandb_run(name="best_model", job_type="best_model") as run:
        results = {
            "test": {}
        }
        models = {}
        for quantile in [0.05, 0.5, 0.95]:
            # train on train + validation split to compute test score
            model = train_lgbm_regressor(
                df=pd.concat([train_df, validation_df], axis=0),
                target=target,
                config=config_quantile,
                alpha=quantile,
            )

            rmse = evaluate_model(
                model, test_df, target="energy_consumption_future_hours_0"
            )
            logger.info(f"[quantile = {quantile}] Test RMSE: {rmse:.2f}")
            logger.info(
                f"[quantile = {quantile}] Test Mean Energy Consumption: {test_df['energy_consumption_future_hours_0'].mean():.2f}"
            )
            results["test"] = {
                f"rmse_{quantile=}": rmse,
            }

            # train on all the dataset for the final model
            model = train_lgbm_regressor(
                df=pd.concat([train_df, validation_df, test_df], axis=0),
                target=target,
                config=config_quantile,
                alpha=quantile,
            )
            models[quantile] = model
            model_path = str(save_model_dir / f"energy_consumption_model_quantile={quantile}.pkl")
            utils.save_model(model, model_path)

            metadata = dict(config_quantile)
            metadata["target"] = target
            metadata["test_rmse"] = rmse
            description = f"""
            LightGBM Regressor trained on the whole dataset with the best hyperparameters found 
            using wandb sweeps as a hyperparamter tuning tool. 
            The model is trained with {quantile=} and {target=}.
            """
            model_artifact = wandb.Artifact(
                f"LGBMRegressorQuantile{quantile}", type=f"model", description=description, metadata=metadata
            )
            # wandb.save(model_path)
            model_artifact.add_file(model_path)

            run.log_artifact(model_artifact)

        wandb.log(results)


def plot_results(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_model_dir: str = "../models",
        target: str = "energy_consumption_future_hours_0",
):
    """
    Visualize results of the models.

    Args:
        train_df:
        validation_df:
        test_df:
        target:

    Returns:

    """

    # Load models
    save_model_dir = Path(save_model_dir)
    models = {}
    for model_path in save_model_dir.iterdir():
        if model_path.suffix == ".pkl":
            quantile = float(model_path.stem.split("=")[-1])
            model = utils.load_model(model_path)
            models[quantile] = model

    # Prepare data for plotting
    data = utils.load_data_from_parquet("../energy_consumption_data.parquet")
    data = data[["datetime_utc", "area", "consumer_type"]]
    train_df_raw, validation_df_raw, test_df_raw = preprocess.split_data(data)

    train_df = pd.concat([train_df, train_df_raw[["area", "consumer_type"]]], axis=1)
    validation_df = pd.concat([validation_df, validation_df_raw[["area", "consumer_type"]]], axis=1)
    test_df = pd.concat([test_df, test_df_raw[["area", "consumer_type"]]], axis=1)

    train_groups = train_df.groupby(["area", "consumer_type"])
    validation_groups = validation_df.groupby(["area", "consumer_type"])
    test_groups = test_df.groupby(["area", "consumer_type"])

    with init_wandb_run(name="results", job_type="results") as run:
        for train_timeseries, validation_timeseries, test_timeseries in zip(train_groups, validation_groups, test_groups):
            train_group_name, train_timeseries = train_timeseries
            validation_group_name, validation_timeseries = validation_timeseries
            test_group_name, test_timeseries = test_timeseries

            assert train_group_name == validation_group_name == test_group_name, \
                f"Group names are not the same: {train_group_name}, {validation_group_name}, {test_group_name}"

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_title(f"Results for {train_group_name} - predicting: {target}")
            ax.plot(train_timeseries["datetime_utc"], train_timeseries[target], label="train", alpha=0.5)
            ax.plot(validation_timeseries["datetime_utc"], validation_timeseries[target], label="validation", alpha=0.5)
            ax.plot(test_timeseries["datetime_utc"], test_timeseries[target], label="test", alpha=0.5)
            for quantile, model in models.items():
                y_pred = model.predict(test_timeseries.drop(columns=[target, "datetime_utc", "area", "consumer_type"]))
                ax.plot(test_timeseries["datetime_utc"], y_pred, label=f"test_{quantile}")

                if quantile != 0.5:
                    ax.fill_between(
                        test_timeseries["datetime_utc"],
                        test_timeseries[target],
                        y_pred,
                        alpha=0.25,
                        color="red"
                    )
            ax.legend()
            wandb.log({"results": wandb.Image(fig)})
            plt.close(fig)


if __name__ == "__main__":
    # train_df, validation_df, test_df = get_dataset(
    #     data_path="../energy_consumption_data.parquet"
    # )
    main()

    # wandb.agent(
    #     project="energy_consumption",
    #     sweep_id=sweep_id,
    #     function=partial(train_sweep, train_df=train_df, validation_df=validation_df),
    # )
    # train_best_model(
    #     train_df=train_df,
    #     validation_df=validation_df,
    #     test_df=test_df,
    # )
    # plot_results(
    #     train_df=train_df,
    #     validation_df=validation_df,
    #     test_df=test_df,
    # )
