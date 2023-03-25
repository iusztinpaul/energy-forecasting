from datetime import timedelta, datetime

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
        requirements=["/opt/airflow/dags/wheels/feature_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False
    )
    def run_feature_pipeline():
        from feature_pipeline import run_feature_pipeline as _run_feature_pipeline

        return _run_feature_pipeline.run()

    @task.virtualenv(
        task_id="create_feature_view",
        requirements=["/opt/airflow/dags/wheels/feature_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        multiple_outputs=False,
        system_site_packages=False
    )
    def create_feature_view():
        from feature_pipeline import create_feature_view as _create_feature_view

        _create_feature_view.run()

    @task.virtualenv(
        task_id="hyperparameter_tuning",
        requirements=["/opt/airflow/dags/wheels/training_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        system_site_packages=False
    )
    def hyperparameter_tuning():
        from training_pipeline import hyperparameter_tuning as _hyperparameter_tuning

        _hyperparameter_tuning.run()

    @task.virtualenv(
        task_id="upload_best_model",
        requirements=["/opt/airflow/dags/wheels/training_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        system_site_packages=False
    )
    def upload_best_model():
        from training_pipeline import upload_best_model as _upload_best_model

        _upload_best_model.run()

    @task.virtualenv(
        task_id="train",
        requirements=["/opt/airflow/dags/wheels/training_pipeline-0.1.0-py3-none-any.whl"],
        python_version="3.9",
        system_site_packages=False
    )
    def train():
        from training_pipeline import train as _train

        _train.run()

    @task.virtualenv(
        task_id="batch_predict",
        requirements=[
            "/opt/airflow/dags/wheels/batch_prediction_pipeline-0.1.0-py3-none-any.whl",
            "/opt/airflow/dags/wheels/training_pipeline-0.1.0-py3-none-any.whl"
        ],
        python_version="3.9",
        system_site_packages=False
    )
    def batch_predict():
        from batch_prediction_pipeline import predict as _predict

        _predict.run()

    # run_feature_pipeline() >> create_feature_view() >> hyperparameter_tuning() >> upload_best_model() >> train() >> batch_predict()
    batch_predict()


feature_pipeline()
