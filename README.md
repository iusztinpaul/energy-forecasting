# The Full Stack 7-Steps MLOps Framework

### LIVE DEMO [FORECASTING](http://35.207.134.188:8501/) | LIVE DEMO [MONITORING](http://35.207.134.188:8502/)

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
5. [Set Up Additional Tools](#tools)
6. [Usage](#usage)
7. [Installation & Usage for Development](#installation)
8. [Licensing & Contributing](#licensing)

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
<br/>
<br/>
<br/>

# 🔧 Set Up Additional Tools <a name=tools></a>

**The code is tested only on Ubuntu 20.04 and 22.04 using Python 3.9.**

If you have problems during the installation, please leave us an issue and we will respond to you and update the README for future readers.

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

[Official Poetry installation instructions.](https://python-poetry.org/docs/#installation)

## Docker

 <br/>[Install Docker on Ubuntu.](https://docs.docker.com/engine/install/ubuntu/) <br/>
[Install Docker on Mac.](https://docs.docker.com/desktop/install/mac-install/) <br/>
[Install Docker on Windows.](https://docs.docker.com/desktop/install/windows-install/) <br/>


## Configure Credentials for the Private PyPi Server
**We will run the private PyPi server using Docker down the line. But it will already expect the credentials configuration.**

Create credentials using `passlib`:
```shell
sudo apt install -y apache2-utils
pip install passlib

mkdir ~/.htpasswd
htpasswd -sc ~/.htpasswd/htpasswd.txt energy-forecasting
```

Set credentials inside `poetry`:
```shell
poetry config repositories.my-pypi http://localhost
poetry config http-basic.my-pypi energy-forecasting <password>
```

Check credentials in your poetry `auth.toml` file:
```shell
cat ~/.config/pypoetry/auth.toml
```

## Hopsworks 

We will use [Hopsworks](https://www.hopsworks.ai/) as our serverless feature store. Thus, you have to create an account and a project on Hopsworks. We will show you later how to configure our code to use your Hopsworks project.

**If you want everything to work with the default settings use the following names:**
- create a `project` called `energy_consumption`

[Click here to start with Hopsworks](https://www.hopsworks.ai/).


## Weights & Biases

We will use Weights & Biases as our serverless ML plaform. Thus, you have to create an account and a project on Weights & Biases. We will show you later how to configure our code to use your W&B project.

**If you want everything to work with the default settings use the following names:**
- create an `entity` called `teaching-mlops`
- create a `project` called `energy_consumption`

[Click here to start with Weights & Biases](https://wandb.ai/).

## GCP

**If you only want to run the code locally follow only the steps from the "Storage" section.**<br/>

As before, you have to create an account and a project. Using solely the bucket as storage will be free of charge.
At the time I am writing this documentation GCS is free until 5GB.

**If you want everything to work with the default settings use the following names:**
- create a `project` called `energy_consumption`

### Storage

At this step you have to do five things:
- create a project
- create a bucket
- create a service account that has admin permissions to the newly created bucket
- create a service account that has read-only permissions to the newly create bucket
- download a JSON key for the newly create service accounts.

[Docs for creating a bucket on GCP.](https://cloud.google.com/storage/docs/creating-buckets)<br/>
[Docs for creating a service account on GCP.](https://cloud.google.com/iam/docs/service-accounts-create)<br/>
[Docs for creating a JSON key for a GCP service account.](https://cloud.google.com/iam/docs/keys-create-delete)<br/>

Your `bucket admin service account` should have assigned the following role: `Storage Object Admin`<br/>
Your `bucket read-only service account` should have assigned the following role: `Storage Object Viewer`<br/>

Again, I want to highligh that at the time I am writing this course GCP storage is free until 5GB.

**If you want everything to work with the default settings use the following names:**
- create a `bucket` called `hourly-batch-predictions`
- rename your `admin` JSON service key to `admin-buckets.json`
- rename your `read-only` JSON service key to `read-buckets.json`

If you want to see more step-by-step instructions checkout this [Medium article](placeholder Medium article).


### Deployment

This step has to be finished only if you want to deploy the code on GCP VMs and build the CI/CD with GitHub Actions.

Note that this step might result in a few consts on GCP. It won't be much. While I developed this course I spent only ~20$. For you most probably it will be less.

Also, if you have a new GCP account, you most probably got some free credits. Just be sure to delete the resources after you finished the course.

See [this document](/README_DEPLOY.md) for detailed instructions on this step.


# 🔎 Usage <a name=usage></a>

**The code is tested only on Ubuntu 20.04 and 22.04 using Python 3.9.**

If you have problems during the installation, please leave us an issue and we will respond to you and update the README for future readers.

## The Pipeline

#### Run 
We will run the pipeline using Airflow. Don't be scared. Docker makes everything very simple to setup.

**NOTE:** We also hooked the **private PyPi server** in the same docker-compose.yaml file. Thus, everythign will start with one command.

```shell
# Move to the airflow directory.
cd airflow

# Make expected directories and environment variables
mkdir -p ./logs ./plugins
sudo chmod 777 ./logs ./plugins

# It will be used by Airflow to identify your user.
echo -e "AIRFLOW_UID=$(id -u)" > .env
# This signals our project root directory
echo "ML_PIPELINE_ROOT_DIR=/opt/airflow/dags" >> .env
```

Now move to the DAGS directory:
```shell
cd ./dags

# Make a copy of the env default file.
cp .env.default .env
# Open the .env file and complete the WANDB_API_KEY and FS_API_KEY credentials 

# Create the folder where the program expects its GCP credentials.
mkdir -p credentials/gcp/energy_consumption
# Copy the GCP service credetials that gives you admin access to GCS. 
cp -r /path/to/admin/gcs/credentials/admin-buckets.json credentials/gcp/energy_consumption
# NOTE that if you want everything to work outside the box your JSON file should be called admin-buckets.json.
# Otherwise you have to manually configure the GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH variable from the .env file. 

# Initialize the Airflow database
docker compose up airflow-init

# Start up all services
# Note: You should setup the private PyPi server credentials before running this command.
docker compose --env-file .env up --build -d
```

Wait a while for the containers to build and run. After access `127.0.0.1:8080` to login into Airflow.<br/>
Use the following credentials to login:
* username: `airflow`
* password: `airflow`

<p align="center">
  <img src="images/airflow_login_screenshot.png">
</p>

Before starting the pipeline DAG, we have one last step to do. Go back to the root folder of the `energy-forecasting` repository and run the following to deploy the pipeline modules to your private PyPi server:
```shell
sh deploy/ml-pipeline.sh
```
Airflow will know how to install the packages from this private PyPi server. <br/>

Now, go to the `DAGS/All` section and search for the `ml_pipeline` DAG. Toggle the activation button. It should automatically start in a few seconds. Also, you can manually run it hitting the play button from the top-right side of the `ml_pipeline` window.

<p align="center">
  <img src="images/airflow_ml_pipeline_dag_overview_screenshot.png">
</p>

That is it. If all the credentials are setup corectly you can run the entire pipeline with a single button. How cool is that?

Here is how the DAG should look like 👇

<p align="center">
  <img src="images/airflow_ml_pipeline_dag_screenshot.png">
</p>


#### Clean Up
```shell
docker compose down --volumes --rmi all
```

#### Run Private PyPi Server Separately

The private Pypi server is already hooked to the airflow docker compose file. But if you want to run it separately for whatever reason you can run this command instead:
```shell
docker run -p 80:8080 -v ~/.htpasswd:/data/.htpasswd pypiserver/pypiserver:latest run -P .htpasswd/htpasswd.txt --overwrite
```

## The Web App

Here, everything is a lot simpler. Here we need to setup only a few credentials. <br/>

Copy the bucket read only GCP credentials to the root directory of your `energy-forecasting` project:
```shell
# Create the folder where the program expects its GCP credentials.
mkdir -p credentials/gcp/energy_consumption
# Copy the GCP service credetials that gives you admin access to GCS. 
cp -r /path/to/admin/gcs/credentials/read-buckets.json credentials/gcp/energy_consumption
# NOTE that if you want everything to work outside the box your JSON file should be called read-buckets.json.
# Otherwise you have to manually configure the APP_API_GCP_SERVICE_ACCOUNT_JSON_PATH variable from the .env file of the API.
```

Go to the API folder and make a copy of the `.env.default` file:
```shell
cd ./app-api
cp .env.default .env
```
**NOTE:** If you set all the names as explain in this README, you shouldn't change anything else.

That is it!

Go back to the root directory of your `energy-forecasting` project and run the following docker command which will build and run all the docker containers of the web app:
```shell
docker compose -f deploy/app-docker-compose.yml --project-directory . up --build
```

If you want to run it in development mode run the following command:
```shell
docker compose -f deploy/app-docker-compose.yml -f deploy/app-docker-compose.local.yml --project-directory . up --build
```

**Now you can see the apps running here:**
* [API](http://127.0.0.1:8001/api/v1/docs)
* [Frontend](http://127.0.0.1:8501/)
* [Monitoring](http://127.0.0.1:8502/)

# 🧑‍💻 Installation & Usage for Development <a name=installation></a>

All the modules support Poetry. Thus the installation is straightforward.

**NOTE:** Make sure that you have installed Python 3.9, not Python 3.8 or Python 3.10.

## The Pipeline

**We support Docker to run the whole pipeline. Check out the [Usage](#usage) section if you only want to run it as a whole.**<br/><br/> 

If Poetry is not using Python 3.9, you can follow the next steps:
1. Install Python 3.9 on your machine.
2. `cd /path/to/project`, for example `cd ./feature-pipeline`
3. run `which python3.9` to find where Python3.9 is
3. run `poetry env use /path/to/python3.9`

##### Set Up the ML_PIPELINE_ROOT_DIR Variable

**!!!** Before installing every module individually, **one key step** is to set the `ML_PIPELINE_ROOT_DIR` variable to your root directory of the `energy-forecasting` project:
```shell
gedit ~/.bashrc
export ML_PIPELINE_ROOT_DIR=/path/to/root/directory/energy-forecasting/repository
```

Another option is to run every Python script with the `ML_PIPELINE_ROOT_DIR` variables. For example:
```shell
ML_PIPELINE_ROOT_DIR=/path/to/root/directory/energy-forecasting/repository python -m feature_pipeline.pipeline
```

-------

**See here how to install every project individually:**
- [Feature Pipeline](/feature-pipeline/README.md)
- [Training Pipeline](/training-pipeline/README.md)
- [Batch Prediction Pipeline](/batch-prediction-pipeline/README.md)


## The Web App
**We support Docker to run the web app. Check out the [Usage](#usage) section if you only want to run it as a whole.**<br/><br/> 

**See here how to install every project individually:**
- [API](/app-api/README.md)
- [Frontend](/app-frontend/README.md)
- [Monitoring](/app-monitoring/README.md)`


# 🏆 Licensing & Contributing <a name=licensing></a>

The code is under the MIT License. Thus, as long as you keep distributing the License, feel free to share, clone, change the code as you like.

Also, if you find any bugs or missing pieces in the documentation I encourage you to add an issue on GitHub. I will take the time to respond you and adapt the code and docs for future readers.

Thanks!
