import datetime
from json import JSONDecodeError
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import requests

from yarl import URL

from feature_pipeline import utils


logger = utils.get_logger(__name__)


def from_api(
    export_end_reference_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour",
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Extract data from the DK energy consumption API.

    IMPORTANT NOTE: This dataset will not be updated in July 2023. The dataset will expire during 2023.
    Here is the link to the dataset: https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour

    Args:
        export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
            Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
            The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].
        days_delay: Data has a delay of N days. Thus, we have to shift our window with N days.
        days_export: The number of days to export.
        url: The URL of the API.

    Returns:
          A tuple of a Pandas DataFrame containing the exported data and a dictionary of metadata.
    """

    # Compute the export window.
    if export_end_reference_datetime is None:
        # As the dataset will expire in July 2023, we set the export end reference datetime to the last day of June 2023 + the delay.
        export_end_reference_datetime = datetime.datetime(
            2023, 6, 30, 23, 0, 0
        ) + datetime.timedelta(days=days_delay)
        export_end_reference_datetime = export_end_reference_datetime.replace(
            minute=0, second=0, microsecond=0
        )
    else:
        export_end_reference_datetime = export_end_reference_datetime.replace(
            minute=0, second=0, microsecond=0
        )

    # TODO: Change the API source, until then we have to clamp the export_end_reference_datetime to the last day of June 2023 to simulate the same behavior.
    expiring_dataset_datetime = datetime.datetime(2023, 6, 30) + datetime.timedelta(
        days=days_delay
    )
    if export_end_reference_datetime > expiring_dataset_datetime:
        export_end_reference_datetime = expiring_dataset_datetime

        logger.warning(
            "We clapped 'export_end_reference_datetime' to 'datetime(2023, 6, 30) + datetime.timedelta(days=days_delay)' as \
        the dataset will not be updated starting from July 2023. The dataset will expire during 2023. \
        Check out the following link for more information: https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour"
        )

    export_end = export_end_reference_datetime - datetime.timedelta(days=days_delay)
    export_start = export_end_reference_datetime - datetime.timedelta(
        days=days_delay + days_export
    )

    # Query API.
    query_params = {
        "offset": 0,
        "sort": "HourUTC",
        "timezone": "utc",
        "start": export_start.strftime("%Y-%m-%dT%H:%M"),
        "end": export_end.strftime("%Y-%m-%dT%H:%M"),
    }
    url = URL(url) % query_params
    url = str(url)
    logger.info(f"Requesting data from API with URL: {url}")
    response = requests.get(url)
    logger.info(f"Response received from API with status code: {response.status_code} ")

    # Parse API response.
    try:
        response = response.json()
    except JSONDecodeError:
        logger.error(
            f"Response status = {response.status_code}. Could not decode response from API with URL: {url}"
        )

        return None

    records = response["records"]
    records = pd.DataFrame.from_records(records)

    # Prepare metadata.
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    metadata = {
        "days_delay": days_delay,
        "days_export": days_export,
        "url": url,
        "export_datetime_utc_start": export_start.strftime(datetime_format),
        "export_datetime_utc_end": export_end.strftime(datetime_format),
        "datetime_format": datetime_format,
        "num_unique_samples_per_time_series": len(records["HourUTC"].unique()),
    }

    return records, metadata
