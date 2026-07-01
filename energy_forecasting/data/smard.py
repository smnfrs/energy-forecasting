"""Low-level SMARD API client.

All functions return DataFrames or raise DataNotAvailableError.
No file I/O -- that's handled by SmardSource in sources.py.
"""

import pandas as pd
import requests
from loguru import logger

from energy_forecasting.config.smard import HTTP_TIMEOUT_SECONDS, SMARD_API_BASE

# Module-level default session for connection pooling.
# SmardSource creates its own session per download/update for thread safety,
# but standalone callers get pooling too via this fallback.
_default_session = requests.Session()


class DataNotAvailableError(Exception):
    """Raised when a SMARD filter/region combination returns 404."""


def get_timestamps(
    filter_id: int,
    region: str,
    resolution: str = "hour",
    session: requests.Session | None = None,
) -> list[int]:
    """Fetch available weekly-chunk timestamps for a filter/region.

    Returns list of timestamps in milliseconds. These are the valid
    chunk IDs for get_data() -- roughly one per week.

    Raises DataNotAvailableError if the combination doesn't exist.
    """
    s = session or _default_session
    url = f"{SMARD_API_BASE}/{filter_id}/{region}/index_{resolution}.json"
    resp = s.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    if resp.status_code == 404:
        raise DataNotAvailableError(f"No data for filter={filter_id}, region={region}")
    resp.raise_for_status()
    return resp.json()["timestamps"]


def get_data(
    filter_id: int,
    region: str,
    timestamp: int,
    resolution: str = "hour",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one weekly chunk of data.

    Returns DataFrame with columns: timestamp (ms), value, time (UTC).
    """
    s = session or _default_session
    url = (
        f"{SMARD_API_BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    resp = s.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()

    series = resp.json()["series"]
    df = pd.DataFrame(series, columns=["timestamp", "value"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def get_all_data(
    filter_id: int,
    region: str,
    resolution: str = "hour",
    timestamp_list: list[int] | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch all data for a filter/region, optionally from specific timestamps.

    If timestamp_list is None, fetches all available timestamps.
    Returns concatenated DataFrame with UTC DatetimeIndex and a single
    'value' column.
    """
    s = session or _default_session
    if timestamp_list is None:
        timestamp_list = get_timestamps(filter_id, region, resolution, session=s)

    if not timestamp_list:
        return pd.DataFrame()

    chunks = []
    for ts in timestamp_list:
        try:
            chunk = get_data(filter_id, region, ts, resolution, session=s)
            chunks.append(chunk)
        except requests.HTTPError:
            logger.warning(f"Failed to fetch chunk {ts} for {filter_id}/{region}")

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["value"])  # Drop unfilled future rows
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("time").sort_index()
    return df[["value"]]
