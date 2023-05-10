# Training Pipeline

## Install for Development

Create virtual environment:
```shell
cd training-pipeline
poetry shell
poetry install
```

Check the [Set Up Additional Tools](https://github.com/iusztinpaul/energy-forecasting#-set-up-additional-tools-) and [Usage](https://github.com/iusztinpaul/energy-forecasting#usage) sections to see **how to set up** the **additional tools** and **credentials** you need to run this project.

Check out this [Medium article](https://medium.com/towards-data-science/a-guide-to-building-effective-training-pipelines-for-maximum-results-6fdaef594cee) for more details about this module.


## Usage for Development

</br> **Run the scripts in the following order:** </br></br>


1. Start the hyperparameter tuning script:
```shell
python -m training_pipeline.hyperparameter_tuning
```

2. Upload the best config based on the previous hyperparameter tuning step:
```shell
python -m training_pipeline.best_config
```
3. Start the training script using the best configuration uploaded one step before:
```shell
python -m training_pipeline.train
```

**NOTE:** Be careful to set the `ML_PIPELINE_ROOT_DIR` variable as explain in this [section](https://github.com/iusztinpaul/energy-forecasting#set-up-the-ml_pipeline_root_dir-variable).
