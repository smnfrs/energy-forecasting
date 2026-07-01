"""Open-Meteo weather data collection.

Three endpoints with different temporal coverage:
1. Archive API -- historical actuals (2015+, hourly only)
2. Historical Forecast API -- forecasts as issued (~2022+, hourly)
3. Forecast API -- current 14-day forecast (hourly)

Each (asset_type, TSO) combination produces three Parquet files:
  weather/{type}/{TSO}/history.parquet
  weather/{type}/{TSO}/hist_forecast.parquet
  weather/{type}/{TSO}/forecast.parquet

Location data loaded from data/locations/eu_locations.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pysolar.solar import get_altitude, get_azimuth

from energy_forecasting.config import LOCATIONS_DIR, WEATHER_DIR
from energy_forecasting.data.io import load_parquet, save_parquet

# ── Variable sets ───────────────────────────────────────────────────
# Ported from EMA's collect_data_openmeteo.py

VARS_BASIC: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
]

VARS_WIND: list[str] = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
]

VARS_RADIATION: list[str] = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
]

# Which variable groups each asset type uses
ASSET_VARIABLES: dict[str, list[str]] = {
    "offshore": VARS_BASIC + VARS_WIND,
    "onshore": VARS_BASIC + VARS_WIND,
    "solar": VARS_BASIC + VARS_RADIATION,
    "cities": VARS_BASIC + VARS_WIND + VARS_RADIATION,
}

# ── Physical limits ─────────────────────────────────────────────────
# Values outside these bounds are clipped. Ported from EMA.
PHYSICAL_LIMITS: dict[str, tuple[float, float]] = {
    "temperature_2m": (-45, 50),
    "relative_humidity_2m": (0, 100),
    "surface_pressure": (900, 1080),
    "precipitation": (0, 100),
    "cloud_cover": (0, 100),
    "wind_speed_10m": (0, 200),
    "wind_speed_100m": (0, 200),
    "wind_direction_10m": (0, 360),
    "wind_direction_100m": (0, 360),
    "wind_gusts_10m": (0, 300),
    "shortwave_radiation": (0, 1400),
    "direct_radiation": (0, 1200),
    "diffuse_radiation": (0, 450),
    "direct_normal_irradiance": (0, 1200),
    "global_tilted_irradiance": (0, 1400),
    "terrestrial_radiation": (200, 2000),
}

# ── API URLs ────────────────────────────────────────────────────────
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Historical forecast API has data from ~2022-01-01
HIST_FORECAST_START = "2022-01-01"


def load_locations(asset_type: str, tso: str) -> list[dict]:
    """Load locations for a given asset type and TSO from eu_locations.json."""
    path = LOCATIONS_DIR / "eu_locations.json"
    with open(path) as f:
        data = json.load(f)

    type_key = asset_type  # keys in eu_locations.json match our asset_type names

    for country in data["countries_metadata"]:
        if country["code"] == "DE":
            return [loc for loc in country["locations"].get(type_key, []) if loc["TSO"] == tso]
    return []


def _validate_physical(variable: str, values: np.ndarray) -> np.ndarray:
    """Clip values to physical limits."""
    if variable in PHYSICAL_LIMITS:
        lo, hi = PHYSICAL_LIMITS[variable]
        return np.clip(values, lo, hi)
    return values


def _add_solar_columns(df: pd.DataFrame, locations: list[dict]) -> pd.DataFrame:
    """Compute solar elevation and azimuth for each location.

    Ported from EMA's add_solar_elevation_and_azimuth().
    Uses pysolar -- computed at collection time so the columns
    are available for feature engineering without re-computation.
    """
    for loc in locations:
        suffix = loc["suffix"]
        lat, lon = loc["lat"], loc["lon"]
        elevations = []
        azimuths = []
        for ts in df.index:
            dt = ts.to_pydatetime()
            elevations.append(get_altitude(lat, lon, dt))
            azimuths.append(get_azimuth(lat, lon, dt))
        df[f"solar_elevation_deg{suffix}"] = elevations
        df[f"solar_azimuth_deg{suffix}"] = azimuths
    return df


def _make_client():
    """Create an openmeteo_requests client with minimal HTTP retry.

    HTTP-level: 1 retry for genuine network glitches only.
    Rate limiting is handled by the app-level retry in _fetch_hourly
    with progressive sleep. Using more HTTP retries here wastes API
    quota because each retry on a rate-limit response counts as a call.
    """
    import openmeteo_requests
    import requests as req_lib
    from retry_requests import retry

    session = retry(req_lib.Session(), retries=1, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


# Application-level retry settings for rate-limit handling.
# EMA used factor=20 (0s, 20s, 40s, 60s, 80s) but the first retry at 0s
# always hits the minutely limit again, wasting a call. Starting at 60s
# gives the minutely limit time to reset.
_MAX_APP_RETRIES = 3
_RETRY_SLEEP_SECONDS = (60, 90)  # sleep durations for attempts 1, 2 (attempt 0 has no sleep)


class RateLimitExhausted(Exception):
    """Raised when Open-Meteo hourly/daily limit is hit and retries won't help."""


def _is_hard_rate_limit(msg: str) -> bool:
    """Detect hourly/daily limits that won't resolve with short retries."""
    lower = msg.lower()
    return "hourly" in lower and "limit" in lower or "daily" in lower and "limit" in lower


def _is_server_error(msg: str) -> bool:
    """Detect server-side failures (504, connection drops) that burn quota without returning data."""
    lower = msg.lower()
    return "504" in lower or "incomplete chunked read" in lower or "peer closed" in lower


def _fetch_hourly(
    url: str,
    params: dict,
    variables: list[str],
    locations: list[dict],
    client=None,
) -> pd.DataFrame:
    """Fetch hourly data from an Open-Meteo endpoint for multiple locations.

    Retry strategy:
    - Rate limits (hourly/daily): raise RateLimitExhausted immediately
    - Server errors (504, connection drops): retry once then give up
      (each attempt burns quota even when it fails)
    - Minutely limits: progressive sleep (20*i seconds), up to 5 attempts

    Raises RateLimitExhausted for hourly/daily limits.
    """
    import time

    if client is None:
        client = _make_client()

    server_errors = 0
    responses = None
    for attempt in range(_MAX_APP_RETRIES):
        try:
            responses = client.weather_api(url, params=params)
            break
        except Exception as exc:
            exc_msg = str(exc)
            if _is_hard_rate_limit(exc_msg):
                raise RateLimitExhausted(exc_msg) from exc
            if _is_server_error(exc_msg):
                server_errors += 1
                if server_errors >= 2:
                    logger.warning(
                        f"Giving up after {server_errors} server errors "
                        f"(each burns quota): {exc}"
                    )
                    raise
            if attempt < _MAX_APP_RETRIES - 1:
                delay = _RETRY_SLEEP_SECONDS[attempt]
                logger.warning(
                    f"Open-Meteo request failed (attempt {attempt + 1}/{_MAX_APP_RETRIES}), "
                    f"retrying in {delay}s: {exc}"
                )
                time.sleep(delay)
            else:
                raise
    hourly_responses = [r.Hourly() for r in responses]

    if len(hourly_responses) != len(locations):
        raise ValueError(f"Expected {len(locations)} responses, got {len(hourly_responses)}")

    result = pd.DataFrame()
    for hourly, loc in zip(hourly_responses, locations):
        time_range = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        data = {}
        for i, var in enumerate(variables):
            col = f"{var}{loc['suffix']}"
            values = hourly.Variables(i).ValuesAsNumpy()
            data[col] = _validate_physical(var, values)

        loc_df = pd.DataFrame(data, index=time_range)

        if result.empty:
            result = loc_df
        else:
            result = result.join(loc_df, how="outer")

    return result


class OpenMeteoSource:
    """Weather data for one (asset_type, TSO) combination.

    Manages three Parquet files: history, hist_forecast, forecast.
    """

    def __init__(
        self,
        asset_type: str,
        tso: str,
        start_date: str = "2015-01-01",
    ):
        self.asset_type = asset_type
        self.tso = tso
        self.start_date = start_date
        self.locations = load_locations(asset_type, tso)
        self.variables = ASSET_VARIABLES[asset_type]
        self.output_dir = WEATHER_DIR / asset_type / tso
        self._client = None  # Lazy-init, reused across endpoint calls

        if not self.locations:
            logger.warning(f"No locations for {asset_type}/{tso}")

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client()
        return self._client

    def _path(self, scope: str) -> Path:
        return self.output_dir / f"{scope}.parquet"

    def download(self) -> None:
        """Full download of all three endpoints.

        Resume-safe: skips endpoints whose parquet file already exists.
        To force re-download, delete the file first.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.locations:
            return

        # 1. Historical actuals (Archive API, 2015+)
        if self._path("history").exists():
            logger.debug(f"Skipping archive (exists): {self.asset_type}/{self.tso}")
        else:
            logger.info(
                f"Fetching archive weather: {self.asset_type}/{self.tso} "
                f"({len(self.locations)} locations)"
            )
            df_actual = self._fetch_archive()
            if not df_actual.empty:
                if self.asset_type == "solar":
                    df_actual = _add_solar_columns(df_actual, self.locations)
                save_parquet(df_actual, self._path("history"))

        # 2. Historical forecasts (2022+)
        if self._path("hist_forecast").exists():
            logger.debug(f"Skipping hist_forecast (exists): {self.asset_type}/{self.tso}")
        else:
            logger.info(f"Fetching historical forecasts: {self.asset_type}/{self.tso}")
            df_hist_fc = self._fetch_historical_forecast()
            if not df_hist_fc.empty:
                if self.asset_type == "solar":
                    df_hist_fc = _add_solar_columns(df_hist_fc, self.locations)
                save_parquet(df_hist_fc, self._path("hist_forecast"))

        # 3. Current forecast (14-day window)
        # Skip if exists during initial download to conserve API quota.
        # Use `update` to refresh forecasts once all history is downloaded.
        if self._path("forecast").exists():
            logger.debug(f"Skipping forecast (exists): {self.asset_type}/{self.tso}")
        else:
            logger.info(f"Fetching current forecast: {self.asset_type}/{self.tso}")
            df_forecast = self._fetch_current_forecast()
            if not df_forecast.empty:
                if self.asset_type == "solar":
                    df_forecast = _add_solar_columns(df_forecast, self.locations)
                save_parquet(df_forecast, self._path("forecast"))

    def update(self) -> None:
        """Incremental update with 3-day overlap.

        - Archive: extend from (last_date - 3 days) to yesterday
        - Historical forecast: extend from (last_date - 3 days) to yesterday
        - Current forecast: replace entirely (it's the latest 14-day window)
        """
        if not self._path("history").exists():
            self.download()
            return

        if not self.locations:
            return

        overlap_days = 3

        # Update archive (actual weather)
        existing = load_parquet(self._path("history"))
        last_ts = existing.dropna(how="all").index.max()
        update_start = (last_ts - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")

        df_new = self._fetch_archive(start_override=update_start)
        if not df_new.empty:
            if self.asset_type == "solar":
                df_new = _add_solar_columns(df_new, self.locations)
            merged = pd.concat([existing, df_new])
            merged = merged[~merged.index.duplicated(keep="last")]
            save_parquet(merged.sort_index(), self._path("history"))

        # Update historical forecast (same overlap pattern)
        if self._path("hist_forecast").exists():
            existing_hf = load_parquet(self._path("hist_forecast"))
            last_ts_hf = existing_hf.dropna(how="all").index.max()
            update_start_hf = (last_ts_hf - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")

            df_new_hf = self._fetch_historical_forecast(start_override=update_start_hf)
            if not df_new_hf.empty:
                if self.asset_type == "solar":
                    df_new_hf = _add_solar_columns(df_new_hf, self.locations)
                merged_hf = pd.concat([existing_hf, df_new_hf])
                merged_hf = merged_hf[~merged_hf.index.duplicated(keep="last")]
                save_parquet(merged_hf.sort_index(), self._path("hist_forecast"))

        # Replace current forecast entirely
        df_forecast = self._fetch_current_forecast()
        if not df_forecast.empty:
            if self.asset_type == "solar":
                df_forecast = _add_solar_columns(df_forecast, self.locations)
            save_parquet(df_forecast, self._path("forecast"))

    def _base_params(self) -> dict:
        """Common parameters for all Open-Meteo endpoints."""
        return {
            "latitude": [loc["lat"] for loc in self.locations],
            "longitude": [loc["lon"] for loc in self.locations],
            "hourly": self.variables,
        }

    def _fetch_archive(self, start_override: str | None = None) -> pd.DataFrame:
        """Fetch from Open-Meteo Archive API."""
        start = start_override or self.start_date
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        params = {**self._base_params(), "start_date": start, "end_date": end, "timezone": "UTC"}
        return _fetch_hourly(ARCHIVE_URL, params, self.variables, self.locations, client=self.client)

    def _fetch_historical_forecast(self, start_override: str | None = None) -> pd.DataFrame:
        """Fetch from Open-Meteo Historical Forecast API (~2022+).

        Fetches in yearly chunks to avoid server timeouts on large requests.
        The hist_forecast endpoint reconstructs forecast data on the fly,
        which is much heavier than the archive endpoint's static data.
        """
        start = pd.Timestamp(start_override or HIST_FORECAST_START, tz="UTC")
        end = pd.Timestamp.now(tz="UTC")

        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.DateOffset(years=1), end)
            logger.debug(
                f"Hist forecast chunk: {chunk_start.strftime('%Y-%m-%d')} "
                f"-> {chunk_end.strftime('%Y-%m-%d')}"
            )
            params = {
                **self._base_params(),
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "timeout": 360,
            }
            chunk_df = _fetch_hourly(
                HIST_FORECAST_URL, params, self.variables, self.locations, client=self.client
            )
            if not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end

        if not chunks:
            return pd.DataFrame()
        result = pd.concat(chunks)
        return result[~result.index.duplicated(keep="last")].sort_index()

    def _fetch_current_forecast(self) -> pd.DataFrame:
        """Fetch current 14-day forecast."""
        params = {**self._base_params(), "forecast_days": 14}
        return _fetch_hourly(FORECAST_URL, params, self.variables, self.locations, client=self.client)
