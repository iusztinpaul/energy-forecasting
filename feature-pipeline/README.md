# Feature Pipeline

Check out [Lesson 1](https://medium.com/towards-data-science/a-framework-for-building-a-production-ready-feature-engineering-pipeline-f0b29609b20f) on Medium to better understand how we built the FE pipeline. 

Also, check out [Lesson 5](https://towardsdatascience.com/ensuring-trustworthy-ml-systems-with-data-validation-and-real-time-monitoring-89ab079f4360) to learn how we implemented the data validation layer using Great Expectations.

## Install for Development

Create virtual environment:
```shell
cd feature-pipeline
poetry shell
poetry install
```

Check the [Set Up Additional Tools](https://github.com/iusztinpaul/energy-forecasting#-set-up-additional-tools-) and [Usage](https://github.com/iusztinpaul/energy-forecasting#usage) sections to see **how to set up** the **additional tools** and **credentials** you need to run this project.

> [!IMPORTANT]  
> If you are on OSX and poetry fails because of `fatal error: 'librdkafka/rdkafka.h' file not found`, you should follow Hopsworks [instructions](https://docs.hopsworks.ai/3.4/user_guides/client_installation/) to install librdkafka:
> ```
> curl -O https://raw.githubusercontent.com/Homebrew/homebrew-core/f7d0f40bbc4075177ecf16812fd95951a723a996/Formula/librdkafka.rb
> brew install --build-from-source librdkafka.rb
> # After librdkafka is installed
> export C_INCLUDE_PATH=$(brew --prefix)/include
> export LIBRARY_PATH=$(brew --prefix)/lib
> poetry install
> ```


## Usage for Development

To start the ETL pipeline run:
```shell
python -m feature_pipeline.pipeline
```

To create a new feature view run:
```shell
python -m feature_pipeline.feature_view
```

**NOTE:** Be careful to complete the `.env` file and set the `ML_PIPELINE_ROOT_DIR` variable as explained in the [Set Up the ML_PIPELINE_ROOT_DIR Variable](https://github.com/iusztinpaul/energy-forecasting#set-up-the-ml_pipeline_root_dir-variable) section of the main README.
