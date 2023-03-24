from datetime import timedelta, datetime

from airflow.decorators import dag, task
from airflow.sensors.external_task import ExternalTaskSensor


@dag(
    dag_id="training_pipeline",
    schedule=timedelta(hours=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["pipeline"]
)
def training_pipeline():
    wait_for_feature_pipeline = ExternalTaskSensor(
        task_id='wait_for_feature_pipeline',
        external_dag_id='feature_pipeline',
        external_task_id='create_feature_view',
        start_date=datetime(2020, 4, 29),
        execution_delta=timedelta(minutes=1),
        check_existence=True
    )

    @task.virtualenv(
        task_id="hyperparameter_tuning",
        requirements=["/opt/airflow/dags/training_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def hyperparameter_tuning():
        from training_pipeline import hyperparameter_tuning as _hyperparameter_tuning

        _hyperparameter_tuning.run()

    @task.virtualenv(
        task_id="upload_best_model",
        requirements=["/opt/airflow/dags/training_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def upload_best_model():
        from training_pipeline import upload_best_model as _upload_best_model

        _upload_best_model.run()

    @task.virtualenv(
        task_id="train",
        requirements=["/opt/airflow/dags/training_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def train():
        from training_pipeline import train as _train

        _train.run()

    wait_for_feature_pipeline >> hyperparameter_tuning() >> upload_best_model() >> train()


training_pipeline()
