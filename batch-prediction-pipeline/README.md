# Batch Prediction Pipeline

## Install for Development

The batch prediction pipeline uses the training pipeline module as a dependency. Thus, as a first step, we must ensure that the training pipeline module is published to our private PyPi server.

**NOTE:** Make sure that your private PyPi server is running. Check the [Usage section](https://github.com/iusztinpaul/energy-forecasting#the-pipeline) if it isn't.

```shell
cd training-pipeline
poetry build
poetry publish -r my-pypi
cd ..
```

```shell
cd batch-prediction-pipeline
poetry shell
poetry install
```

Check the [Set Up Additional Tools](https://github.com/iusztinpaul/energy-forecasting#-set-up-additional-tools-) and [Usage](https://github.com/iusztinpaul/energy-forecasting#usage) sections to see **how to set up** the **additional tools** and **credentials** you need to run this project.
