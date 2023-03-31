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
**TODO:** Move the installation to poetry. Do I need it as this code is running directly in Airflow?
```shell
pip install "apache-airflow[celery]==2.5.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.5.2/constraints-3.7.txt"
```

#### Run
```shell
# Move to the airflow directory.
cd airflow

# Download the docker-compose.yaml file
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'

# Make expected directories and set an expected environment variable
mkdir -p ./dags ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Initialize the database
docker-compose up airflow-init

# Start up all services
docker-compose up --build
```

#### Clean Up
```shell
docker compose down --volumes --rmi all
```



# Private PyPi Server

### Publish Modules
Set experimental installer to false:
```shell
poetry config experimental.new-installer false
```
Create credentials:
```shell
sudo apt install apache2-utils
pip install passlib

mkdir ~/.htpasswd
htpasswd -sc ~/.htpasswd/htpasswd.txt energy-forecasting
```
Set credentials:
```shell
poetry config repositories.test http://localhost
poetry config http-basic.test energy-forecasting <password>
```
Check credentials:
```shell
 cat ~/.config/pypoetry/auth.toml
```
Build and publish:
```shell
cd <module>
poetry build
poetry publish -r test
```

### Run Server
```shell
docker run -p 80:8080 -v ~/.htpasswd:/data/.htpasswd pypiserver/pypiserver:latest run -P .htpasswd/htpasswd.txt --overwrite
```