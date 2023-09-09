import datetime
from typing import Optional
import fire
import pandas as pd

from feature_pipeline.etl import cleaning, load, extract, validation
from feature_pipeline import utils

logger = utils.get_logger(__name__)


def run(
    export_end_reference_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://drive.google.com/uc?export=download&id=1y48YeDymLurOTUO-GeFOUXVNc9MCApG5",
    feature_group_version: int = 1,
) -> dict:
    """
    Extract data from the API, transform it, and load it to the feature store.

    As the official API expired in July 2023, we will use a copy of the data to simulate the same behavior. 
    We made a copy of the data between '2020-06-30 22:00' and '2023-06-30 21:00'. Thus, there are 3 years of data to play with.

    Here is the link to the official obsolete dataset: https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour
    Here is the link to the copy of the dataset: https://drive.google.com/file/d/1y48YeDymLurOTUO-GeFOUXVNc9MCApG5/view?usp=drive_link

    Args:
        export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
            Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
            The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].
        days_delay: Data has a delay of N days. Thus, we have to shift our window with N days.
        days_export: The number of days to export.
        url: The URL of the API or of the copy of the data stored on GitHub.
        feature_group_version: The version of the feature store feature group to save the data to.

    Returns:
          A dictionary containing metadata of the pipeline.
    """

    logger.info(f"Extracting data from API.")
    data, metadata = extract.from_file(
        export_end_reference_datetime, days_delay, days_export, url
    )
    if metadata["num_unique_samples_per_time_series"] < days_export * 24:
        raise RuntimeError(
            f"Could not extract the expected number of samples from the api: {metadata['num_unique_samples_per_time_series']} < {days_export * 24}. \
            Check out the API at: https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour "
        )
    logger.info("Successfully extracted data from API.")

    logger.info(f"Transforming data.")
    data = transform(data)
    logger.info("Successfully transformed data.")

    logger.info("Building validation expectation suite.")
    validation_expectation_suite = validation.build_expectation_suite()
    logger.info("Successfully built validation expectation suite.")

    logger.info(f"Validating data and loading it to the feature store.")
    load.to_feature_store(
        data,
        validation_expectation_suite=validation_expectation_suite,
        feature_group_version=feature_group_version,
    )
    metadata["feature_group_version"] = feature_group_version
    logger.info("Successfully validated data and loaded it to the feature store.")

    logger.info(f"Wrapping up the pipeline.")
    utils.save_json(metadata, file_name="feature_pipeline_metadata.json")
    logger.info("Done!")

    return metadata


def transform(data: pd.DataFrame):
    """
    Wrapper containing all the transformations from the ETL pipeline.
    """

    data = cleaning.rename_columns(data)
    data = cleaning.cast_columns(data)
    data = cleaning.encode_area_column(data)

    return data


if __name__ == "__main__":
    fire.Fire(run)
