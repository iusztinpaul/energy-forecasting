# Feature Pipeline

## Install for Development

Run:
```shell
cd feature-pipeline
poetry shell
poetry install
```

Check the [Set Up Additional Tools](https://github.com/iusztinpaul/energy-forecasting#-set-up-additional-tools-) and [Usage](https://github.com/iusztinpaul/energy-forecasting#usage) sections to see **how to set up** the **additional tools** and **credentials** you need to run this project.


## Usage for Development

To start the ETL pipeline run:
```shell
python -m feature_pipeline.pipeline
```

To create a new feature view run:
```shell
python -m feature_pipeline.feature_view
```

Check out this [Medium article](https://medium.com/towards-data-science/a-framework-for-building-a-production-ready-feature-engineering-pipeline-f0b29609b20f) for more details about this module.
