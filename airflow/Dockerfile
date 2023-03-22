FROM apache/airflow:2.5.2

RUN pip install poetry

ADD pyproject.toml .
#ADD poetry.lock .
RUN poetry install --no-dev
