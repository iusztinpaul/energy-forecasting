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
        export_end_reference_datetime = datetime.datetime.utcnow().replace(
            minute=0, second=0, microsecond=0
        )
    else:
        export_end_reference_datetime = export_end_reference_datetime.replace(
            minute=0, second=0, microsecond=0
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
    }

    return records, metadata
