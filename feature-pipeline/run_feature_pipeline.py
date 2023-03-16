import fire
import pandas as pd

import cleaning
import extract
import load
import utils
import validation


logger = utils.get_logger(__name__)


def main(
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour",
) -> dict:
    logger.info(f"Extracting data from API.")
    data, metadata = extract.from_api(days_delay, days_export, url)
    logger.info("Successfully extracted data from API.")

    logger.info(f"Transforming data.")
    data = transform(data)
    logger.info("Successfully transformed data.")

    logger.info(f"Validating data and loading it to the feature store.")
    load.to_feature_store(
        data,
        validation_expectation_suite=validation.expectation_suite_energy_consumption,
    )
    logger.info("Successfully validated data and loaded it to the feature store.")

    # TODO: Clean the old data from thea feature store to keep the freemium version of Hopsworks.

    logger.info(f"Wrapping up the pipeline.")
    utils.save_json(metadata, file_name="feature_pipeline_metadata.json")
    logger.info("Done!")

    return metadata


def transform(data: pd.DataFrame):
    # Clean columns
    data = cleaning.rename_columns(data)

    # Cast columns
    data = cleaning.cast_columns(data)

    # Standardize categorical data
    data = cleaning.standardize_categorical_data(data)

    return data


if __name__ == "__main__":
    fire.Fire(main)
