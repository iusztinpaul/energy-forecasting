import datetime
from json import JSONDecodeError
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import requests

from yarl import URL

import utils


logger = utils.get_logger(__name__)


def from_api(
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://api.energidataservice.dk/dataset/ConsumptionDE35Hour",
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Extract data from the API.

    Args:
        days_delay: Data has a delay of N days. Thus, we have to shift our window with N days.
        days_export: The number of days to export.
        url: The URL of the API.

    Returns:
          A tuple of a Pandas DataFrame containing the exported data and a dictionary of metadata.
    """

    current_datetime = datetime.datetime.utcnow().replace(
        minute=0, second=0, microsecond=0
    )
    export_start = current_datetime - datetime.timedelta(days=days_delay + days_export)
    export_start = export_start.strftime("%Y-%m-%dT%H:%M")

    query_params = {
        "offset": 0,
        "sort": "HourUTC",
        "timezone": "utc",
        "start": export_start,
    }
    url = URL(url) % query_params
    url = str(url)
    logger.info(f"Requesting data from API with URL: {url}")

    response = requests.get(url)
    try:
        response = response.json()
    except JSONDecodeError:
        logger.error(
            f"Response status = {response.status_code}. Could not decode response from API with URL: {url}"
        )

        return None

    records = response["records"]
    records = pd.DataFrame.from_records(records)

    # standardize datetime format
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    export_start = datetime.datetime.strptime(export_start, "%Y-%m-%dT%H:%M")
    export_start = export_start.strftime(datetime_format)
    export_end = datetime.datetime.strptime(
        records["HourUTC"].max(), "%Y-%m-%dT%H:%M:%S"
    )
    export_end = export_end.strftime(datetime_format)

    metadata = {
        "days_delay": days_delay,
        "days_export": days_export,
        "url": url,
        "export_datetime_utc_start": export_start,
        "export_datetime_utc_end": export_end,
        "datetime_format": datetime_format,
    }

    return records, metadata
