from datetime import timedelta, datetime

from airflow.decorators import dag, task


from feature_pipeline import run_feature_pipeline as _run_feature_pipeline
from feature_pipeline import create_feature_view as _create_feature_view


@dag(
    schedule=timedelta(hours=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["feature-pipeline"]
)
def feature_pipeline():
    @task(multiple_outputs=True)
    def run_feature_pipeline():
        return _run_feature_pipeline.run()

    @task()
    def create_feature_view():
        _create_feature_view.run()

    run_feature_pipeline()
    create_feature_view()


feature_pipeline()
