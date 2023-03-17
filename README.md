# energy-forecasting

# Data
We used the daily energy consumption from Denmark data which you can access [here](https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour).


# Pipelines 
## #1. Feature Engineering Pipeline

## #2. Training Pipeline

## #3. Batch Prediction Pipeline

-----

# Orchestration

## Airflow

### Setup
You can read the official documentation [here](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) or follow the steps bellow for a fast start.

**TODO:** This setup is used for development. Check out what I have to do for production.

#### Install Python Package
**TODO:** Move the installation to poetry. Do I need it as this code is ran directly in Airflow?
```shell
pip install "apache-airflow[celery]==2.5.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.5.2/constraints-3.7.txt"
```

#### Prepare structure
```shell
cd airflow
mkdir -p ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

#### Initialize Database
```shell
docker compose up airflow-init
```

#### Run
```shell
docker compose up
```

#### Clean Up
```shell
docker compose down --volumes --rmi all
```
