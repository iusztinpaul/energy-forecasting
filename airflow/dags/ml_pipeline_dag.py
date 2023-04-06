from datetime import timedelta, datetime

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.models.baseoperator import chain
        

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
        system_site_packages=True
    )
    def run_feature_pipeline(
        days_delay: int,
        days_export: int,
        url: str,
        feature_group_version: int,
    ):
        from feature_pipeline import utils, run_feature_pipeline as _run_feature_pipeline

        logger = utils.get_logger(__name__)
        
        logger.info(f"days_delay = {days_delay}")
        logger.info(f"days_export = {days_export}")
        logger.info(f"url = {url}")
        logger.info(f"feature_group_version = {feature_group_version}")

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
    def hyperparameter_tuning(feature_view_metadata: dict, run: bool):
        from training_pipeline import utils, hyperparameter_tuning as _hyperparameter_tuning

        logger = utils.get_logger(__name__)

        if run is True:
            logger.info("Running hyperparameter tuning.")

            return _hyperparameter_tuning.run(
                feature_view_version=feature_view_metadata["feature_view_version"],
                training_dataset_version=feature_view_metadata["training_dataset_version"]
            )
        else:
            logger.info("Skipping hyperparameter tuning.")

            return {}

    @task.virtualenv(
        task_id="upload_best_model",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "training_pipeline"],
        python_version="3.9",
        multiple_outputs=False,
        system_site_packages=False
    )
    def upload_best_model(last_sweep_metadata: dict):
        from training_pipeline import utils, upload_best_model as _upload_best_model

        logger = utils.get_logger(__name__)

        sweep_id = last_sweep_metadata.get("sweep_id")
        if sweep_id is not None:
            logger.info("Uploading best config from the sweep.")

            _upload_best_model.run(
                sweep_id=last_sweep_metadata.get("sweep_id")
            )
            has_best_config = True
        else:
            logger.info("No sweep found. Checking if a best config exists.")

            has_best_config = utils.check_if_artifact_exists("best_config")

        return has_best_config
    
    @task.short_circuit
    def should_run_hyperparameter_tuning(run_hyperparameter_tuning: bool) -> bool:
        return run_hyperparameter_tuning
    
    @task.short_circuit
    def should_not_run_hyperparameter_tuning(run_hyperparameter_tuning: bool) -> bool:
        return run_hyperparameter_tuning


    @task.virtualenv(
        task_id="train",
        requirements=["--trusted-host 172.17.0.1", "--extra-index-url http://172.17.0.1", "training_pipeline"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def train(feature_view_metadata: dict, has_best_config: bool):
        from training_pipeline import train as _train

        if has_best_config is False:
            raise RuntimeError("No best config found. Please run hyperparameter tuning first.")

        return _train.run(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
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

    days_delay = int(Variable.get("days_delay", default_var=15))
    days_export = int(Variable.get("days_export", default_var=30))
    url = Variable.get("url", default_var="https://api.energidataservice.dk/dataset/ConsumptionDE35Hour")
    feature_group_version = int(Variable.get("feature_group_version", default_var=1))
    run_hyperparameter_tuning = bool(Variable.get("run_hyperparameter_tuning", default_var=False))

    feature_pipeline_metadata = run_feature_pipeline(
        days_delay=days_delay,
        days_export=days_export,
        url=url,
        feature_group_version=feature_group_version
        )
    feature_view_metadata = create_feature_view(feature_pipeline_metadata)

    last_sweep_metadata = hyperparameter_tuning(feature_view_metadata, run=run_hyperparameter_tuning)
    found_best_config = upload_best_model(last_sweep_metadata)
    train_metadata = train(feature_view_metadata, found_best_config)
    batch_predict_operation = batch_predict(feature_view_metadata, train_metadata)
    
    chain(should_run_hyperparameter_tuning(run_hyperparameter_tuning), last_sweep_metadata, found_best_config, train_metadata, batch_predict_operation)
    chain(should_not_run_hyperparameter_tuning(run_hyperparameter_tuning), train_metadata, batch_predict_operation)



feature_pipeline()
