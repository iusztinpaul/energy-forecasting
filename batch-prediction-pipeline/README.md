# Batch Prediction Pipeline

Check out [Lesson 3](https://towardsdatascience.com/unlock-the-secret-to-efficient-batch-prediction-pipelines-using-python-a-feature-store-and-gcs-17a1462ca489) on Medium to better understand how we built the batch prediction pipeline. 

Also, check out [Lesson 5](https://towardsdatascience.com/ensuring-trustworthy-ml-systems-with-data-validation-and-real-time-monitoring-89ab079f4360) to learn how we implemented the monitoring layer to compute the model's real-time performance.

## Install for Development

The batch prediction pipeline uses the training pipeline module as a dependency. Thus, as a first step, we must ensure that the training pipeline module is published to our private PyPi server.

**NOTE:** Make sure that your private PyPi server is running. Check the [Usage section](https://github.com/iusztinpaul/energy-forecasting#the-pipeline) if it isn't.

Build & publish the `training-pipeline` to your private PyPi server:
```shell
cd training-pipeline
poetry build
poetry publish -r my-pypi
cd ..
```

Install the virtual environment for `batch-prediction-pipeline`:
```shell
cd batch-prediction-pipeline
poetry shell
poetry install
```

Check the [Set Up Additional Tools](https://github.com/iusztinpaul/energy-forecasting#-set-up-additional-tools-) and [Usage](https://github.com/iusztinpaul/energy-forecasting#usage) sections to see **how to set up** the **additional tools** and **credentials** you need to run this project.

## Usage for Development

To start batch prediction script, run:
```shell
python -m batch_prediction_pipeline.batch
```

To compute the monitoring metrics based, run the following:
```shell
python -m batch_prediction_pipeline.monitoring
```

**NOTE:** Be careful to complete the `.env` file and set the `ML_PIPELINE_ROOT_DIR` variable as explained in the [Set Up the ML_PIPELINE_ROOT_DIR Variable](https://github.com/iusztinpaul/energy-forecasting#set-up-the-ml_pipeline_root_dir-variable) section of the main README.
