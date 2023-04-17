import datetime
from json import JSONDecodeError
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import requests

from yarl import URL

from feature_pipeline import utils


logger = utils.get_logger(__name__)


def from_api(
    export_end_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour",
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Extract data from the DK energy consumption API.

    Args:
        export_reference_end_datetime: The end datetime of the export window. If None, the current time is used.
            Note that if the difference between the current datetime and the "export_end_datetime" is less than the "days_delay",
            the "export_start_datetime" is shifted accordingly to ensure we extract a number of "days_export" days.
        days_delay: Data has a delay of N days. Thus, we have to shift our window with N days.
        days_export: The number of days to export.
        url: The URL of the API.

    Returns:
          A tuple of a Pandas DataFrame containing the exported data and a dictionary of metadata.
    """

    # Query API.
    if export_end_datetime is None:
        export_end = datetime.datetime.utcnow().replace(
            minute=0, second=0, microsecond=0
        )
    else:
        export_end = export_end_datetime.replace(minute=0, second=0, microsecond=0)

    # Compute the potential delay we still have to add to the export window relative to the give "export_end_datetime".
    remained_days_delay = datetime.timedelta(days=days_delay) - (
        datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        - export_end
    )
    remained_days_delay = max(remained_days_delay.days, 0)

    export_start = export_end - datetime.timedelta(
        days=remained_days_delay + days_export
    )

    export_end = export_end.strftime("%Y-%m-%dT%H:%M")
    export_start = export_start.strftime("%Y-%m-%dT%H:%M")

    query_params = {
        "offset": 0,
        "sort": "HourUTC",
        "timezone": "utc",
        "start": export_start,
        "end": export_end,
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
    export_start = datetime.datetime.strptime(export_start, "%Y-%m-%dT%H:%M")
    export_start = export_start.strftime(datetime_format)
    export_end = datetime.datetime.strptime(export_end, "%Y-%m-%dT%H:%M")
    export_end = export_end.strftime(datetime_format)

    metadata = {
        "days_delay": days_delay,
        "remained_days_delay": remained_days_delay,
        "days_export": days_export,
        "url": url,
        "export_datetime_utc_start": export_start,
        "export_datetime_utc_end": export_end,
        "datetime_format": datetime_format,
    }

    return records, metadata
