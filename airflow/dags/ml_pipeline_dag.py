from datetime import timedelta, datetime
from typing import Optional

from airflow.decorators import dag, task


@dag(
    dag_id="ml-pipeline",
    schedule=timedelta(hours=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"]
)
def feature_pipeline():
    @task.virtualenv(
        task_id="run_feature_pipeline",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "feature_pipeline"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def run_feature_pipeline(
        days_delay: int = 15,
        days_export: int = 30,
        url: str = "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour",
        feature_group_version: int = 2
    ):
        from feature_pipeline import run_feature_pipeline as _run_feature_pipeline

        return _run_feature_pipeline.run(
            days_delay=days_delay,
            days_export=days_export,
            url=url,
            feature_group_version=feature_group_version
        )

    @task.virtualenv(
        task_id="create_feature_view",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "feature_pipeline"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def create_feature_view(feature_pipeline_metadata: dict):
        from feature_pipeline import create_feature_view as _create_feature_view

        return _create_feature_view.run(
            feature_group_version=feature_pipeline_metadata["feature_group_version"]
        )

    @task.virtualenv(
        task_id="hyperparameter_tuning",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "training_pipeline"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def hyperparameter_tuning(feature_view_metadata: dict):
        from training_pipeline import hyperparameter_tuning as _hyperparameter_tuning

        return _hyperparameter_tuning.run(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"]
        )

    @task.virtualenv(
        task_id="upload_best_model",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "training_pipeline"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def upload_best_model(last_sweep_metadata: dict):
        from training_pipeline import upload_best_model as _upload_best_model

        return _upload_best_model.run(
            sweep_id=last_sweep_metadata["sweep_id"]
        )

    @task.virtualenv(
        task_id="train",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "training_pipeline"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def train(feature_view_metadata: dict, best_model_metadata: dict):
        from training_pipeline import train as _train

        return _train.run(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
            best_model_artifact=best_model_metadata["artifact"]
        )

    @task.virtualenv(
        task_id="batch_predict",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "batch_prediction_pipeline"],
        python_version="3.9",
        system_site_packages=False
    )
    def batch_predict(feature_view_metadata: dict, train_metadata: dict, fh: int = 24):
        from batch_prediction_pipeline import predict as _predict

        _predict.run(
            fh=fh,
            feature_view_version=feature_view_metadata["feature_view_version"],
            model_version=train_metadata["model_version"]
        )

    feature_pipeline_metadata = run_feature_pipeline()
    feature_view_metadata = create_feature_view(feature_pipeline_metadata)
    last_sweep_metadata = hyperparameter_tuning(feature_view_metadata)
    best_model_metadata = upload_best_model(last_sweep_metadata)
    train_metadata = train(feature_view_metadata, best_model_metadata)
    batch_predict(feature_view_metadata, train_metadata)


feature_pipeline()
