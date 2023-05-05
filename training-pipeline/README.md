# Training Pipeline

## Install for Development

Run:
```shell
cd training-pipeline
poetry shell
poetry install
```

Check the [Set Up Additional Tools](https://github.com/iusztinpaul/energy-forecasting#-set-up-additional-tools-) and [Usage](https://github.com/iusztinpaul/energy-forecasting#usage) sections to see **how to set up** the **additional tools** and **credentials** you need to run this project.


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
3. Start the training script based on the best configuration uploaded one step before:
```shell
python -m training_pipeline.train
```

Check out this [Medium article](placeholder-medium-article) for more details about this module.


**NOTE:** Be careful to set the `ML_PIPELINE_ROOT_DIR` variable as explain in this [section](https://github.com/iusztinpaul/energy-forecasting#set-up-the-ml_pipeline_root_dir-variable).
