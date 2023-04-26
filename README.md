# energy-forecasting

[LIVE DEMO](http://35.207.134.188:8501/)

# Data
We used the daily energy consumption from Denmark data which you can access [here](https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour).

# Pipelines 
## #1. Feature Engineering Pipeline

## #2. Training Pipeline

## #3. Batch Prediction Pipeline

-----

# Install


# Setup Machine

## Poetry
Install Python system dependencies:
```shell
sudo apt-get install -y python3-distutils
```

```shell
curl -sSL https://install.python-poetry.org | python3 -
nano ~/.bashrc
```
Add `export PATH=~/.local/bin:$PATH` to `~/.bashrc`
Check poetry
```shell
source ~/.bashrc
poetry --version
```

Check official Poetry instructions [here](https://python-poetry.org/docs/#installation).

## Private PyPi Server Credentials
Install pip:
```shell
sudo apt-get -y install python3-pip
```

Create credentials:
```shell
sudo apt install -y apache2-utils
pip install passlib

mkdir ~/.htpasswd
htpasswd -sc ~/.htpasswd/htpasswd.txt energy-forecasting
```
Set credentials:
```shell
poetry config repositories.my-pypi http://localhost
poetry config http-basic.my-pypi energy-forecasting <password>
```
Check credentials:
```shell
cat ~/.config/pypoetry/auth.toml
```


## Airflow & Private PyPi Server

Run backfil:
```shell
docker exec -it <container_id_of_airflow-airflow-webserver> sh
# Rerun runs:
airflow tasks clear --start-date "2023/04/11 00:00:00" --end-date "2023/04/13 23:59:59" ml_pipeline
# Run from scratch:
airflow dags backfill --start-date "2023/04/11 00:00:00" --end-date "2023/04/13 23:59:59" ml_pipeline
```

## Install Docker

Install Docker on GCP instructions [here](https://tomroth.com.au/gcp-docker/).

TLDR
```shell
sudo apt update
sudo apt install --yes apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt update
sudo apt install --yes docker-ce
```
docker sudo access:
```shell
sudo usermod -aG docker $USER
logout 
```

### Setup
You can read the official documentation [here](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) or follow the steps bellow for a fast start.

#### Run

clone repo
```shell
git clone https://github.com/iusztinpaul/energy-forecasting.git
cd energy-forecasting
```

```shell
# Move to the airflow directory.
cd airflow

# Make expected directories and set an expected environment variable
mkdir -p ./logs ./plugins
sudo chmod 777 ./logs ./plugins
echo -e "AIRFLOW_UID=$(id -u)" > .env
echo "ML_PIPELINE_ROOT_DIR=/opt/airflow/dags" >> .env



cd ./dags
# Copy bucket writing access GCP service JSON file
mkdir -p credentials/gcp/energy_consumption
gcloud compute scp --recurse --zone europe-west3-c --quiet --tunnel-through-iap --project silver-device-379512 ~/Documents/credentials/gcp/energy_consumption/admin-buckets.json ml-pipeline:~/energy-forecasting/airflow/dags/credentials/gcp/energy_consumption/

touch .env
# Complete env vars from the .env file
# Check .env.default for all possible variables.
gcloud compute scp --recurse --zone europe-west3-c --quiet --tunnel-through-iap --project silver-device-379512 ~/Documents/projects/energy-forecasting/airflow/dags/.env ml-pipeline:~/energy-forecasting/airflow/dags/

# Initialize the database
cd <airflow_dir>
docker compose up airflow-init

# Start up all services
# Note: You should setup the PyPi server credentials before running the docker containers.
docker compose --env-file .env up --build -d
```

#### Clean Up
```shell
docker compose down --volumes --rmi all
```

#### Set Variables
ml_pipeline_days_export = 30
ml_pipeline_feature_group_version = 5
ml_pipeline_should_run_hyperparameter_tuning = False

## Build & Publish Python Modules
Set experimental installer to false:
```shell
poetry config experimental.new-installer false
```
Build and publish:
```shell
cd <module>
poetry build
poetry publish -r my-pypi
```
Run the following to build and publish all the modules:
```shell
cd <root_dir>
sh deploy/ml-pipeline.sh
```
**NOTE:** Be sure that are modules are deployed before starting the DAG. Otherwise, it won't know how to load them inside the DAG. 

### Run Server
Note that the image is hooked to the airflow docker compose command.
```shell
docker run -p 80:8080 -v ~/.htpasswd:/data/.htpasswd pypiserver/pypiserver:latest run -P .htpasswd/htpasswd.txt --overwrite
```

### GCP

install gcp SDK:
```shell
```

IAM principals - service accounts:
* read-buckets
* admin-buckets
* admin-vm

Firewall rules:
* IAP for TCP tunneling [docs](https://cloud.google.com/iap/docs/using-tcp-forwarding)
* Expose port 8080

[Open 8080 Port](https://stackoverflow.com/questions/21065922/how-to-open-a-specific-port-such-as-9090-in-google-compute-engine)
[Open 8080 Port](https://www.howtogeek.com/devops/how-to-open-firewall-ports-on-a-gcp-compute-engine-instance/)


VM machine:
* 2 vCPU cores - 8 GB RAM / e2-standard-2 with 20 GB Storage

create VM machine `ml-pipeline`:
```shell
```

connect to VM machine through shh:
```shell
gcloud compute ssh ml-pipeline --zone europe-west3-c --quiet --tunnel-through-iap --project silver-device-379512
```
###### Set credentials for GitHub Actions

print json credentials in one line:
```shell
jq -c . admin-vm.json 
```

set private key:
```
```



# Run APP
Copy the GCP credentials with which you can read from the GCP buckets:
```shell
mkdir -p credentials/gcp/energy_consumption
cp your/location/file.json credentials/gcp/energy_consumption/
```
Create `.env` file:
```shell
cp app-api/.env.default app-api/.env
# Change values in .env if necessary
```
Build & run:
```shell
docker compose -f deploy/app-docker-compose.yml --project-directory . up --build
```
Run local dev from root dir:
```shell
docker compose -f deploy/app-docker-compose.yml -f deploy/app-docker-compose.local.ym --project-directory . up --build
```


### Deploy APP on GCP

#### GCP Resources

- VM: e2-micro - 0.25 2 vCPU - 1 GB memory - 15 GB standard persisted disk
- firewall: expose ports 8501, 8001
- firewall: IAP for TCP tunneling
- create static external IP address - [docs](https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address#console)
- service roles for: reading buckets & SSH access

#### Commands

Connect to VM:
```shell
gcloud compute ssh app --zone europe-west3-c --quiet --tunnel-through-iap --project silver-device-379512
```
Install requirements:
```shell
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install -y git
```
Install docker:
```shell
sudo apt update
sudo apt install --yes apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt update
sudo apt install --yes docker-ce

# docker sudo access:
sudo usermod -aG docker $USER
logout
```
SSH again:
```shell
gcloud compute ssh app --zone europe-west3-c --quiet --tunnel-through-iap --project silver-device-379512
```
Clone repo:
```shell
git clone https://github.com/iusztinpaul/energy-forecasting.git
cd energy-forecasting
```
Create credentials folder:
```shell
mkdir -p credentials/gcp/energy_consumption
```
Copy GCP credentials JSON file:
```shell
gcloud compute scp --recurse --zone europe-west3-c --quiet --tunnel-through-iap --project silver-device-379512 ~/Documents/credentials/gcp/energy_consumption/read-buckets.json app:~/energy-forecasting/credentials/gcp/energy_consumption/
```
Create `.env` file:
```shell
cp app-api/.env.default app-api/.env
```
Install numpy to speed up IAP TCP upload bandwidth:
```shell
$(gcloud info --format="value(basic.python_location)") -m pip install numpy
```
Build & run:
```shell
docker compose -f deploy/app-docker-compose.yml --project-directory . up --build
```
