# The Full Stack 7-Steps MLOps Framework

##### LIVE DEMO [FORECASTING](http://35.207.134.188:8501/) | LIVE DEMO [MONITORING](http://35.207.134.188:8502/)

--------

This repository is a **7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model forecasting energy consumption for the next 24 hours across multiple consumer types from Denmark.

This course targets mid/advanced machine learning engineers who want to level up their skills by building their own end-to-end projects.

Following the documentation and the Medium articles you can reproduce and understand every piece of the code!

**At the end of the course you will know how to build everything from the diagram below.**

Don't worry if something doesn't make sense to you. I will explain everything in detail in my Medium series [placeholder for Medium link].

<p align="center">
  <img src="images/architecture.png">
</p>

As long as you keep the LICENSE, you can safely use this code as a starting point for your awesome project.

# Table of Contents
1. [What You will Learn](#learn)
2. [Lessons & Tutorials](#lessons)
3. [Data](#data)
4. [Code Structure](#structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Licensing & Contributing](#licensing)

# 🤔 What You Will Learn <a name=learn></a>
**At the end of this 7 lessons course, you will know how to:**
* design a batch-serving architecture
* use Hopsworks as a feature store
* design a feature engineering pipeline that reads data from an API
* build a training pipeline with hyper-parameter tunning
* use W&B as an ML Platform to track your experiments, models, and metadata
* implement a batch prediction pipeline
* use Poetry to build your own Python packages
* deploy your own private PyPi server
* orchestrate everything with Airflow
* use the predictions to code a web app using FastAPI and Streamlit
* use Docker to containerize your code
* use Great Expectations to ensure data validation and integrity
* monitor the performance of the predictions over time
* deploy everything to GCP
* build a CI/CD pipeline using GitHub Actions

If that sounds like a lot, don't worry, after you will  cover this course you will understand everything I said before. Most importantly, you will know WHY I used all these tools and how they work together as a system.

[placeholder for Medium link to Lesson 1]

# 🤌 Lessons & Tutorials <a name=lessons></a>
**👇 Access the step-by-step lessons on Medium 👇**
1. Batch Serving. Feature Stores. Feature Engineering Pipelines.
2. Training Pipelines. ML Platforms. Hyperparameter Tuning.
3. Batch Prediction Pipeline. Package Python Modules with Poetry.
4. Private PyPi Server. Orchestrate Everything with Airflow.
5. Build Your Own App with FastAPI and Streamlit.
6. Data Validation and Integrity using GE. Monitor Model Performance.
7. Deploy Everything on GCP. Build a CI/CD Pipeline using GitHub Actions.


# 📊 Data <a name=data></a>
We used an open API that provides hourly energy consumption values for all the energy consumer types within Denmark.

They provide an intuitive interface where you can easily query and visualize the data. You can access the data [here](https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour).

The data has 4 main attributes:
* **Hour UTC**: the UTC datetime when the data point was observed. 
* **Price Area**: Denmark is divided into two price areas: DK1 and DK2 - divided by the Great Belt. DK1 is west of the Great Belt, and DK2 is east of the Great Belt.
* **Consumer Type**: The consumer type is the Industry Code DE35, owned and maintained by Danish Energy.
* **Total Consumption**: Total electricity consumption in kWh

**Note:** The observations have a lag of 15 days! But for our demo use case, that is not a problem, as we can simulate the same steps as it would be in real-time.

<p align="center">
  <img src="images/forecasting_demo_screenshot.png">
</p>

The data points have an hourly resolution. For example: "2023–04–15 21:00Z", "2023–04–15 20:00Z", "2023–04–15 19:00Z", etc.

We will model the data as multiple time series. Each unique price area and consumer type tuple represents its unique time series. 

Thus, we will build a model that independently forecasts the energy consumption for the next 24 hours for every time series.

[Check out our live demo to better understand how the data looks.](http://35.207.134.188:8501/)

# 🧬 Code Structure <a name=structure></a>

The code is split in two main components: the pipeline and the web app.

The **pipeline** consists of three modules:
- `feature-pipeline`
- `training-pipeline`
- `batch-prediction-pipeline`

The **web app** consits of other three modules:
- `app-api`
- `app-frontend`
- `app-monitoring`

**Also,** we have the following folders:
- `airflow` : Airflow files | Orchestration
- `.github` : GitHub Actions files | CI/CD


# 🪛 Installation <a name=installation></a>

**The code is tested only on Ubuntu 20.04 and 22.04.**

## Common

### Poetry

Install Python system dependencies:
```shell
sudo apt-get install -y python3-distutils
```

```shell
curl -sSL https://install.python-poetry.org | python3 -
nano ~/.bashrc
```

Add `export PATH=~/.local/bin:$PATH` to `~/.bashrc`

Check if Poetry is intalled:
```shell
source ~/.bashrc
poetry --version
```

[Official Poetry installation instructions.](https://python-poetry.org/docs/#installation)

### Docker

[Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).


### Configure Private PyPi Server
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








# 🔎 Usage <a name=usage></a>

# 🏆 Licensing & Contributing <a name=licensing></a>


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

Check if Poetry is intalled:
```shell
source ~/.bashrc
poetry --version
```


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
docker compose -f deploy/app-docker-compose.yml -f deploy/app-docker-compose.local.yml --project-directory . up --build
```


### Deploy APP on GCP

#### GCP Resources

- VM: e2-micro - 0.25 2 vCPU - 1 GB memory - 15 GB standard persisted disk
- firewall: expose ports 8501, 8502, 8001
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
