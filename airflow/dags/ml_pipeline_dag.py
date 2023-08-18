from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.edgemodifier import Label


@dag(
    dag_id="ml_pipeline",
    schedule="@hourly",
    start_date=datetime(2023, 4, 14),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1,
)
def ml_pipeline():
    @task.virtualenv(
        task_id="run_feature_pipeline",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "feature_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=True,
    )
    def run_feature_pipeline(
        export_end_reference_datetime: str,
        days_delay: int,
        days_export: int,
        url: str,
        feature_group_version: int,
    ) -> dict:
        """
        Run the feature pipeline.

        Args:
            export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
                Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
                The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].

            days_delay : int
                Data has a delay of N days. Thus, we have to shift our window with N days.

            days_export : int
                The number of days to export.

            url : str
                URL to the raw data.

            feature_group_version : int
                Version of the feature store feature group to use.

        Returns:
            Metadata of the feature pipeline run.
        """

        from datetime import datetime

        from feature_pipeline import utils, pipeline

        logger = utils.get_logger(__name__)

        try:
            export_end_reference_datetime = datetime.strptime(
                export_end_reference_datetime, "%Y-%m-%d %H:%M:%S.%f%z"
            )
        except ValueError:
            export_end_reference_datetime = datetime.strptime(
                export_end_reference_datetime, "%Y-%m-%d %H:%M:%S%z"
            )
        export_end_reference_datetime = export_end_reference_datetime.replace(
            microsecond=0, tzinfo=None
        )

        logger.info(f"export_end_datetime = {export_end_reference_datetime}")
        logger.info(f"days_delay = {days_delay}")
        logger.info(f"days_export = {days_export}")
        logger.info(f"url = {url}")
        logger.info(f"feature_group_version = {feature_group_version}")

        return pipeline.run(
            export_end_reference_datetime=export_end_reference_datetime,
            days_delay=days_delay,
            days_export=days_export,
            url=url,
            feature_group_version=feature_group_version,
        )

    @task.virtualenv(
        task_id="create_feature_view",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "feature_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def create_feature_view(feature_pipeline_metadata: dict) -> dict:
        """
        This function creates a feature view based on the feature pipeline computations. The feature view
        is created using the feature group version from the feature pipeline metadata.
        """

        from feature_pipeline import feature_view

        return feature_view.create(
            feature_group_version=feature_pipeline_metadata["feature_group_version"]
        )

    @task.virtualenv(
        task_id="run_hyperparameter_tuning",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "training_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def run_hyperparameter_tuning(feature_view_metadata: dict) -> dict:
        """
        This function runs hyperparameter tuning for the training pipeline.
        The feature store feature view version and training dataset version are passed
        based on the results from the create_feature_view task.
        """

        from training_pipeline import hyperparameter_tuning

        return hyperparameter_tuning.run(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
        )

    @task.virtualenv(
        task_id="upload_best_config",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "training_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=False,
        system_site_packages=False,
    )
    def upload_best_config(last_sweep_metadata: dict):
        """
        Upload the best config to W&B ML platform found in the hyperparameter tuning step
        based on the given sweep id.
        """

        from training_pipeline import best_config

        best_config.upload(sweep_id=last_sweep_metadata["sweep_id"])

    @task.virtualenv(
        task_id="train_from_best_config",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "training_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    def train_from_best_config(feature_view_metadata: dict) -> dict:
        """Trains model from the best config found in hyperparameter tuning.

        Args:
            feature_view_metadata (dict): Contains feature store feature view and training dataset version.

        Returns:
            metadata from the training run
        """

        from training_pipeline import utils, train

        has_best_config = utils.check_if_artifact_exists("best_config")
        if has_best_config is False:
            raise RuntimeError(
                "No best config found. Please run hyperparameter tuning first."
            )

        return train.from_best_config(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
        )

    @task.virtualenv(
        task_id="compute_monitoring",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "batch_prediction_pipeline",
        ],
        python_version="3.9",
        system_site_packages=False,
    )
    def compute_monitoring(feature_view_metadata: dict):
        """Compute monitoring metrics for newly obbserved data.

        Args:
            feature_view_metadata: metadata containing the version of the feature store feature view version.
        """

        from batch_prediction_pipeline import monitoring

        monitoring.compute(
            feature_view_version=feature_view_metadata["feature_view_version"],
        )

    @task.virtualenv(
        task_id="batch_predict",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "batch_prediction_pipeline",
        ],
        python_version="3.9",
        system_site_packages=False,
    )
    def batch_predict(
        feature_view_metadata: dict,
        train_metadata: dict,
        feature_pipeline_metadata: dict,
        fh: int = 24,
    ):
        """
        This is the function that runs the batch prediction pipeline

        Args:
            feature_view_metadata (dict):  the metadata from the create feature view task
            train_metadata (dict): the metadata from the training pipeline task
            feature_pipeline_metadata (dict): the metadata from the feature pipeline task
            fh (int, optional): forecast horizon. Defaults to 24.
        """

        from datetime import datetime
        from batch_prediction_pipeline import batch

        start_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
        end_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

        batch.predict(
            fh=fh,
            feature_view_version=feature_view_metadata["feature_view_version"],
            model_version=train_metadata["model_version"],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    @task.branch(task_id="if_run_hyperparameter_tuning_branching")
    def if_run_hyperparameter_tuning_branching(run_hyperparameter_tuning: bool) -> bool:
        """Task used to branch between hyperparameter tuning and skipping it."""
        if run_hyperparameter_tuning is True:
            return ["branch_run_hyperparameter_tuning"]
        else:
            return ["branch_skip_hyperparameter_tuning"]

    # Define empty operators used for branching between hyperparameter tuning and skipping it.
    branch_run_hyperparameter_tuning_operator = EmptyOperator(
        task_id="branch_run_hyperparameter_tuning"
    )
    branch_skip_hyperparameter_tuning_operator = EmptyOperator(
        task_id="branch_skip_hyperparameter_tuning"
    )

    # Define Airflow variables.
    days_delay = int(Variable.get("ml_pipeline_days_delay", default_var=15))
    days_export = int(Variable.get("ml_pipeline_days_export", default_var=30))
    url = Variable.get(
        "ml_pipeline_url",
        default_var="https://media.githubusercontent.com/media/iusztinpaul/energy-forecasting/main/data/ConsumptionDE35Hour.csv",
    )
    feature_group_version = int(
        Variable.get("ml_pipeline_feature_group_version", default_var=1)
    )
    should_run_hyperparameter_tuning = (
        Variable.get(
            "ml_pipeline_should_run_hyperparameter_tuning", default_var="False"
        )
        == "True"
    )

    # Feature pipeline
    feature_pipeline_metadata = run_feature_pipeline(
        export_end_reference_datetime="{{ dag_run.logical_date }}",
        days_delay=days_delay,
        days_export=days_export,
        url=url,
        feature_group_version=feature_group_version,
    )
    feature_view_metadata = create_feature_view(feature_pipeline_metadata)

    # Training pipeline
    if_run_hyperparameter_tuning_branch = if_run_hyperparameter_tuning_branching(
        should_run_hyperparameter_tuning
    )
    last_sweep_metadata = run_hyperparameter_tuning(feature_view_metadata)
    upload_best_model_step = upload_best_config(last_sweep_metadata)
    train_metadata = train_from_best_config(feature_view_metadata)

    # Batch prediction pipeline
    compute_monitoring_step = compute_monitoring(feature_view_metadata)
    batch_predict_step = batch_predict(
        feature_view_metadata, train_metadata, feature_pipeline_metadata
    )

    # Define DAG structure.
    (
        feature_view_metadata
        >> if_run_hyperparameter_tuning_branch
        >> [
            if_run_hyperparameter_tuning_branch
            >> Label("Run HPO")
            >> branch_run_hyperparameter_tuning_operator
            >> last_sweep_metadata
            >> upload_best_model_step,
            if_run_hyperparameter_tuning_branch
            >> Label("Skip HPO")
            >> branch_skip_hyperparameter_tuning_operator,
        ]
        >> train_metadata
        >> compute_monitoring_step
        >> batch_predict_step
    )


ml_pipeline()
