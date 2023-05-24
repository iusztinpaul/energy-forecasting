# The Full Stack 7-Steps MLOps Framework

### 🔥 LIVE DEMO 🔥 | [WEB APP - FORECASTING](http://35.207.134.188:8501/) | [WEB APP - MONITORING](http://35.207.134.188:8502/)

--------

This repository is a **7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model forecasting energy consumption for the next 24 hours across multiple consumer types from Denmark.

This course targets mid/advanced machine learning engineers who want to level up their skills by building their own end-to-end projects.

Following the documentation and the [Medium articles](#lessons), you can reproduce and understand every piece of the code!

**At the end of the course, you will know how to build everything from the diagram below.**

Don't worry if something doesn't make sense to you. I will explain everything in detail in the [Medium series](#lessons).

More concretely, you will learn how to build, train, serve, and monitor an ML model using a batch architecture. 

<p align="center">
  <img src="images/architecture.png">
</p>

You can safely use this code as you like, as long as you respect the terms and agreement of the MIT License.

``<<< Using all the tools suggested in the course will be free of charge, except the ones from Lesson 7 (last lesson), where you will be deploying your application to GCP which will cost you ~20$. >>>``

# Table of Contents
1. [What You Will Learn](#learn)
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

If that sounds like a lot, don't worry. After you cover this course, you will understand everything I said before. Most importantly, you will know WHY I used all these tools and how they work together as a system.

# 🤌 Lessons & Tutorials <a name=lessons></a>

The course consists of 7 lessons hosted on Medium Towards Data Science publication. To get the best out of this course, you should also run the code while you read the articles.

**👇 Access the step-by-step lessons on Medium 👇**
1. [Batch Serving. Feature Stores. Feature Engineering Pipelines.](https://medium.com/towards-data-science/a-framework-for-building-a-production-ready-feature-engineering-pipeline-f0b29609b20f).
2. [Training Pipelines. ML Platforms. Hyperparameter Tuning.](https://medium.com/towards-data-science/a-guide-to-building-effective-training-pipelines-for-maximum-results-6fdaef594cee)
3. [Batch Prediction Pipeline. Package Python Modules with Poetry.](https://medium.com/towards-data-science/unlock-the-secret-to-efficient-batch-prediction-pipelines-using-python-a-feature-store-and-gcs-17a1462ca489)
4. [Private PyPi Server. Orchestrate Everything with Airflow.](https://towardsdatascience.com/unlocking-mlops-using-airflow-a-comprehensive-guide-to-ml-system-orchestration-880aa9be8cff)
5. Build Your Own App with FastAPI and Streamlit.
6. Data Validation and Integrity using GE. Monitor Model Performance.
7. Deploy Everything to GCP. Build a CI/CD Pipeline using GitHub Actions.


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

The code is split into two main components: the `pipeline` and the `web app`.

The **pipeline** consists of 3 modules:
- `feature-pipeline`
- `training-pipeline`
- `batch-prediction-pipeline`

The **web app** consists of other 3 modules:
- `app-api`
- `app-frontend`
- `app-monitoring`

**Also,** we have the following folders:
- `airflow` : Airflow files | Orchestration
- `.github` : GitHub Actions files | CI/CD
- `deploy` : Build & Deploy
<br/>
<br/>

To follow the structure in its natural flow, read the folders in the following order:
1. `feature-pipeline`
2. `training-pipeline`
3. `batch-prediction-pipeline`
4. `airflow`
5. `app-api`
6. `app-frontend` & `app-monitoring`
7. `.github`

**Read the Medium articles listed in the [Lessons & Tutorials](#lessons) section for the whole experience.**
<br/>
<br/>

# 🔧 Set Up Additional Tools <a name=tools></a>

**The code is tested only on Ubuntu 20.04 and 22.04 using Python 3.9.**

If you have problems during the setup, please leave us an issue, and we will respond to you and update the README for future readers.

Also, if you have any questions, you can contact me directly on [LinkedIn](https://www.linkedin.com/in/pauliusztin/).

## Poetry
##### ``<< free usage >>``

Install Python system dependencies:
```shell
sudo apt-get install -y python3-distutils
```
Download and install Poetry:
```shell
curl -sSL https://install.python-poetry.org | python3 -
```
Open the `.bashrc` file to add the Poetry PATH: 
```shell
nano ~/.bashrc
```

Add `export PATH=~/.local/bin:$PATH`

to `~/.bashrc`

Check if Poetry is installed:
```shell
source ~/.bashrc
poetry --version
```

[Official Poetry installation instructions.](https://python-poetry.org/docs/#installation)

## Docker
##### ``<< free usage >>``

 <br/>[Install Docker on Ubuntu.](https://docs.docker.com/engine/install/ubuntu/) <br/>
[Install Docker on Mac.](https://docs.docker.com/desktop/install/mac-install/) <br/>
[Install Docker on Windows.](https://docs.docker.com/desktop/install/windows-install/) <br/>


## Configure Credentials for the Private PyPi Server
##### ``<< free usage >>``

**<br/>We will run the private PyPi server using Docker down the line. But it will already expect the credentials configured.<br/>**

Create credentials using `passlib`:
```shell
# Install dependencies.
sudo apt install -y apache2-utils
pip install passlib

# Create the credentials under the energy-forecasting name.
mkdir ~/.htpasswd
htpasswd -sc ~/.htpasswd/htpasswd.txt energy-forecasting
```

Set `poetry` to use the credentials:
```shell
poetry config repositories.my-pypi http://localhost
poetry config http-basic.my-pypi energy-forecasting <password>
```

Check that the credentials are set correctly in your poetry `auth.toml` file:
```shell
cat ~/.config/pypoetry/auth.toml
```

## Hopsworks 
##### ``<< free usage >>``

You will use [Hopsworks](https://www.hopsworks.ai/) as your serverless feature store. Thus, you have to create an account and a project on Hopsworks. We will show you how to configure the code to use your Hopsworks project later.

[I explained in this lesson how to create an API Key on Hopsworks.](https://medium.com/towards-data-science/a-framework-for-building-a-production-ready-feature-engineering-pipeline-f0b29609b20f) But long story short, you can go to your Hopsworks account settings and create the API Key from there.

**If you want everything to work with the default settings, use the following naming conventions:**
- create a `project` called `energy_consumption`

[Click here to start with Hopsworks](https://www.hopsworks.ai/).

**Note:** Our course will use only the Hopsworks freemium plan, making it free of charge to replicate the code within the series. 


## Weights & Biases
##### ``<< free usage >>``

You will use Weights & Biases as your serverless ML platform. Thus, you must create an account and a project on Weights & Biases. We will show you how to configure the code to use your W&B project later.

[I explained in this lesson how to create an API Key on W&B.](placeholder Medium article) But long story short, you can go to your W&B user settings and create the API Key from there.

**If you want everything to work with the default settings, use the following naming conventions:**
- create an `entity` called `teaching-mlops`
- create a `project` called `energy_consumption`

[Click here to start with Weights & Biases](https://wandb.ai/).

**Note:** Our course will use only the W&B freemium plan, making it free of charge to replicate the code within the series. 

## GCP

First, you must install the `gcloud` GCP CLI on your machine.

[Follow this tutorial to install it.](https://cloud.google.com/sdk/docs/install)

**If you only want to run the code locally, go straight to the "Storage" section.**<br/>

As before, you have to create an account and a project on GCP. Using solely the bucket as storage will be free of charge.

When I am writing this documentation, GCS is free until 5GB.

**If you want everything to work with the default settings, use the following naming conventions:**
- create a `project` called `energy_consumption`

### Storage
##### ``<< free usage >>``

At this step, you have to do 5 things:
- create a project
- create a non-public bucket
- create a service account that has admin permissions to the newly created bucket
- create a service account that has read-only permissions to the newly created bucket
- download a JSON key for the newly created service accounts.

![Bucket Creation](images/gcp_gcs_screenshot.png)

[Docs for creating a bucket on GCP.](https://cloud.google.com/storage/docs/creating-buckets)<br/>
[Docs for creating a service account on GCP.](https://cloud.google.com/iam/docs/service-accounts-create)<br/>
[Docs for creating a JSON key for a GCP service account.](https://cloud.google.com/iam/docs/keys-create-delete)<br/>

Your `bucket admin service account` should have assigned the following role: `Storage Object Admin`<br/>
Your `bucket read-only service account` should have assigned the following role: `Storage Object Viewer`<br/>


**NOTE:** When I write this course, GCP storage is free until 5GB.

**If you want everything to work with the default settings, use the following naming conventions:**
- create a `project` called `energy_consumption`
- create a `non-public bucket` called `hourly-batch-predictions` (Pick any region, but just be aware of it.)
- rename your downloaded `admin` JSON service key to `admin-buckets.json`
- rename your downloaded `read-only` JSON service key to `read-buckets.json`

Check out our [Medium article](placeholder Medium article) for more step-by-step instructions.


### Deployment
##### ``<< ~20$ >>``

This step must only be finished if you want to deploy the code on GCP VMs and build the CI/CD with GitHub Actions.

Note that this step might result in a few costs on GCP. It won't be much. While developing this course, I spent only ~20$, which will probably be less for you.

Also, you can get some free credits if you create a new GCP account (I had a new account and received 300$ in GCP credits). Just be sure to delete the resources after you finish the course.

See [this document](/README_DEPLOY.md) for detailed instructions.


# 🔎 Usage <a name=usage></a>

**The code is tested only on Ubuntu 20.04 and 22.04 using Python 3.9.**

If you have problems during the usage instructions, please leave us an issue, and we will respond to you and update the README for future readers.

Also, if you have any questions, you can contact me directly on [LinkedIn](https://www.linkedin.com/in/pauliusztin/).

## The Pipeline

Check out this [lesson on Medium](https://towardsdatascience.com/unlocking-mlops-using-airflow-a-comprehensive-guide-to-ml-system-orchestration-880aa9be8cff) to understand how everything is orchestrated using Airflow. 

#### Run 
You will run the pipeline using Airflow (`free usage`). Don't be scared. Docker makes everything very simple to set up.

**NOTE:** We also hooked the **private PyPi server** in the same docker-compose.yaml file with Airflow. Thus, everything will start with one command.

```shell
# Move to the airflow directory.
cd airflow

# Make expected directories and environment variables
mkdir -p ./logs ./plugins
sudo chmod 777 ./logs ./plugins

# It will be used by Airflow to identify your user.
echo -e "AIRFLOW_UID=$(id -u)" > .env
# This shows where our project root directory is located.
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
# Otherwise, you have to manually configure the GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH variable from the .env file. 

# Initialize the Airflow database
docker compose up airflow-init

# Start up all services
# Note: You should set up the private PyPi server credentials before running this command.
docker compose --env-file .env up --build -d
```

[Read the official Airflow installation using Docker, but NOTE that we modified their official docker-compose.yaml file.](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)

Wait a while for the containers to build and run. After access `127.0.0.1:8080` to login into Airflow.<br/>
Use the following default credentials to log in:
* username: `airflow`
* password: `airflow`

<p align="center">
  <img src="images/airflow_login_screenshot.png">
</p>

Before starting the pipeline DAG, you must deploy the modules to the private PyPi server. Go back to the root folder of the `energy-forecasting` repository and run the following to build and deploy the pipeline modules to your private PyPi server:
```shell
# Set the experimental installer of Poetry to False. For us, it crashed when it was on True.
poetry config experimental.new-installer false
# Build & deploy the pipelines modules.
sh deploy/ml-pipeline.sh
```
Airflow will know how to install the packages from the private PyPi server. <br/>

One final step is to configure the parameters used to run the pipeline. Go to the `Admin` tab, then hit `Variables.` There you can click on the `blue` `+` button to add a new variable.
These are the three parameters you can configure with our suggested values:
* `ml_pipeline_days_export = 30`
* `ml_pipeline_feature_group_version = 5`
* `ml_pipeline_should_run_hyperparameter_tuning = False`
<br/>

<p align="center">
  <img src="images/airflow_variables_screenshot.png">
</p>

Now, go to the `DAGS/All` section and search for the `ml_pipeline` DAG. Toggle the activation button. It should automatically start in a few seconds. Also, you can manually run it by hitting the play button from the top-right side of the `ml_pipeline` window.

<p align="center">
  <img src="images/airflow_ml_pipeline_dag_overview_screenshot.png">
</p>

That is it. You can run the entire pipeline with a single button if all the credentials are set up correctly. How cool is that?

Here is what the DAG should look like 👇

<p align="center">
  <img src="images/airflow_ml_pipeline_dag_screenshot.png">
</p>


#### Clean Up
```shell
docker compose down --volumes --rmi all
```

#### Backfil Using Airflow

Find your `airflow-webserver` docker container ID:
```shell
docker ps
```
Start a shell inside the `airflow-webserver` container and run `airflow dags backfill` as follows (in this example, we did a backfill between `2023/04/11 00:00:00` and `2023/04/13 23:59:59`):
```shell
docker exec -it <container-id-of-airflow-webserver> sh
airflow dags backfill --start-date "2023/04/11 00:00:00" --end-date "2023/04/13 23:59:59" ml_pipeline
```
If you want to clear the tasks and run them again, run these commands:
```shell
docker exec -it <container-id-of-airflow-webserver> sh
airflow tasks clear --start-date "2023/04/11 00:00:00" --end-date "2023/04/13 23:59:59" ml_pipeline
```


### Run Private PyPi Server Separately

The private PyPi server is already hooked to the airflow docker compose file. But if you want to run it separately for whatever reason, you can run this command instead:
```shell
docker run -p 80:8080 -v ~/.htpasswd:/data/.htpasswd pypiserver/pypiserver:latest run -P .htpasswd/htpasswd.txt --overwrite
```

## The Web App

Here, everything is a lot simpler. This time, we need to set up only a few credentials. <br/>

Copy the bucket read-only GCP credentials to the root directory of your `energy-forecasting` project:
```shell
# Create the folder where the program expects its GCP credentials.
mkdir -p credentials/gcp/energy_consumption
# Copy the GCP service credetials that gives you read-only access to GCS. 
cp -r /path/to/admin/gcs/credentials/read-buckets.json credentials/gcp/energy_consumption
# NOTE that if you want everything to work outside the box your JSON file should be called read-buckets.json.
# Otherwise, you have to manually configure the APP_API_GCP_SERVICE_ACCOUNT_JSON_PATH variable from the .env file of the API.
```

Go to the API folder and make a copy of the `.env.default` file:
```shell
cd ./app-api
cp .env.default .env
```
**NOTE:** You shouldn't change anything else if you respect all the naming conventions suggested in this README.

That is it!

Go back to the root directory of your `energy-forecasting` project and run the following docker command, which will build and run all the docker containers of the web app:
```shell
docker compose -f deploy/app-docker-compose.yml --project-directory . up --build
```

If you want to run it in development mode, run the following command:
```shell
docker compose -f deploy/app-docker-compose.yml -f deploy/app-docker-compose.local.yml --project-directory . up --build
```

**Now you can see the apps running at:**
* [API](http://127.0.0.1:8001/api/v1/docs)
* [Frontend](http://127.0.0.1:8501/)
* [Monitoring](http://127.0.0.1:8502/)

# 🧑‍💻 Installation & Usage for Development <a name=installation></a>

All the modules support Poetry. Thus the installation is straightforward.

**NOTE:** Just ensure you have installed Python 3.9, not Python 3.8 or Python 3.10.

## The Pipeline

**We support Docker to run the whole pipeline. Check out the [Usage](#usage) section if you only want to run it as a whole.**<br/><br/> 

If Poetry is not using Python 3.9, you can follow the next steps:
1. Install Python 3.9 on your machine.
2. `cd /path/to/project`, for example, `cd ./feature-pipeline`
3. run `which python3.9` to find where Python3.9 is located
4. run `poetry env use /path/to/python3.9`

### Set Up the ML_PIPELINE_ROOT_DIR Variable

**!!!** Before installing every module individually, **one key step** is to set the `ML_PIPELINE_ROOT_DIR` variable to your root directory of the `energy-forecasting` project:
```shell
gedit ~/.bashrc
export ML_PIPELINE_ROOT_DIR=/path/to/root/directory/energy-forecasting/repository
```

Another option is to run every Python script with the `ML_PIPELINE_ROOT_DIR` variables. For example:
```shell
ML_PIPELINE_ROOT_DIR=/path/to/root/directory/energy-forecasting/repository python -m feature_pipeline.pipeline
```

**NOTE:** This step is **critical** as every pipeline component needs to access the previous pipelines' metadata. By setting up the **ML_PIPELINE_ROOT_DIR** variable, all the metadata will be saved and accessed from the same location between different processes. For example, the batch prediction pipeline will read the model version it needs to use to make predictions, which is a byproduct of the training pipeline passed as metadata between processes.

## Deploy the Code to GCP

[Check out this section.](./README_DEPLOY.md)

## Set UP CI/CD with GitHub Actions

[Check out this section.](./README_CICD.md)

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
- [Monitoring](/app-monitoring/README.md)

You can also run the whole web app in development mode using Docker:
```shell
docker compose -f deploy/app-docker-compose.yml -f deploy/app-docker-compose.local.yml --project-directory . up --build
```


# 🏆 Licensing & Contributing <a name=licensing></a>

The code is under the MIT License. Thus, as long as you keep distributing the License, feel free to share, clone, or change the code as you like.

Also, if you find any bugs or missing pieces in the documentation, I encourage you to add an issue on GitHub. I will respond to you and adapt the code and docs for future readers.

Furthermore, you can contact me directly on [LinkedIn](https://www.linkedin.com/in/pauliusztin/) if you have any questions.

-----

I want to thank [Kurtis Pykes](https://github.com/kurtispykes) for being an awesome copilot and helping me make this course happen.
