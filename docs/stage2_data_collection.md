## Stage 2: Data Collection & Storage

**Goal:** All data sources download correctly and produce valid Parquet files. `make data` runs end-to-end. `make update` incrementally updates all sources.

**Source material:**
- EP: `src/data/sources.py` (DataSource base class, SmardSource, IcapSource, YahooSource, FredSource, EnergyChartsSource), `src/data/smard.py` (SMARD API), `src/data/commodities.py` (TTF reconstruction, carbon dual-source, bias correction), `src/config/smard.py`, `src/config/commodities.py`, `src/cli.py`
- EMA: `data_collection_modules/collect_data_smard_v2.py` (per-TSO SMARD), `data_collection_modules/collect_data_openmeteo.py` (three-endpoint weather), `data_collection_modules/parquet_operations.py`, `data_collection_modules/eu_locations.py` (already extracted to `data/locations/eu_locations.json`), `update_database.py`

---

### 2.1 Dependencies

Add weather-collection and testing dependencies that were deferred from stage 1:

```toml
# In pyproject.toml [project] dependencies, add:

    # Weather data (Open-Meteo)
    "openmeteo-requests>=1.3",
    "requests-cache>=1.2",
    "retry-requests>=2.0",

# In pyproject.toml [project.optional-dependencies] dev, add:

    "responses>=0.25",          # HTTP mocking for SMARD/ICAP/EnergyCharts tests
```

`openmeteo-requests` wraps the Open-Meteo API, `requests-cache` avoids re-fetching during development, `retry-requests` handles transient failures. `responses` is the HTTP mocking library used in stage 2 tests.

---

### 2.2 Config: SMARD API Mappings

**`energy_forecasting/config/smard.py`** — SMARD API constants: region codes, TSO mappings, per-TSO filter keys, known missing combinations. Ported from EP's `src/config/smard.py` and EMA's `collect_data_smard_v2.py`.

```python
"""SMARD API region, TSO, and filter key configuration.

Column name mappings (filter key -> column name) are in config/columns.py.
This module handles the API-specific configuration: which regions exist,
which filter keys are valid per region, and known-missing combinations.
"""

# ── Resolution map ──────────────────────────────────────────────────
# SMARD API resolution string -> expected records per day
RESOLUTION_PERIODS: dict[str, int] = {
    "quarterhour": 96,
    "hour": 24,
}

# ── National region codes ───────────────────────────────────────────
# Used in SMARD API URLs. EP downloads from these.
NATIONAL_REGIONS: list[str] = [
    "DE-LU",          # Current bidding zone (Oct 2018+)
    "DE-AT-LU",       # Historical (pre-Oct 2018)
]

# ── TSO region codes ───────────────────────────────────────────────
# Per-TSO SMARD data. EMA downloads from these for gen/load models.
TSO_REGIONS: dict[str, str] = {
    "50Hertz":    "50Hertz",
    "Amprion":    "Amprion",
    "TenneT":     "TenneT",
    "TransnetBW": "TransnetBW",
    "Creos":      "Creos",
}

# TSO suffix for column names (from EMA's eu_locations)
TSO_SUFFIXES: dict[str, str] = {
    "50Hertz":    "_50hz",
    "Amprion":    "_ampr",
    "TenneT":     "_tenn",
    "TransnetBW": "_tran",
    "Creos":      "_lu",
}

# ── Per-TSO filter keys ────────────────────────────────────────────
# These are the generation/load filters available at TSO level.
# Ported from EMA's collect_data_smard_v2.py.
TSO_FILTER_KEYS: dict[int, str] = {
    1225: "wind_offshore",
    4067: "wind_onshore",
    4068: "solar",
    410:  "load",
    4066: "biomass",
    4071: "gas",
    4069: "hard_coal",
    1223: "lignite",
    4070: "pumped_storage",
    1226: "hydro",          # run_of_river + water_reservoir combined
    1227: "other_conv",     # oil + other_fossil
    1228: "other_renew",    # other_renewables + geothermal + waste
}

# ── Known missing combinations ──────────────────────────────────────
# (filter_id, region) pairs that return 404 from the API.
# Ported from EMA's KNOWN_MISSING.
KNOWN_MISSING: set[tuple[int, str]] = {
    # No offshore wind in landlocked/partial TSOs
    (1225, "Amprion"),
    (1225, "TransnetBW"),
    (1225, "Creos"),
    # No lignite in some TSOs
    (1223, "TransnetBW"),
    (1223, "Creos"),
    # No hard coal in Creos
    (4069, "Creos"),
    # No pumped storage in Creos
    (4070, "Creos"),
    # No other renewables in Creos
    (1228, "Creos"),
}

# ── SMARD API base URL ─────────────────────────────────────────────
SMARD_API_BASE = "https://smard.api.proxy.bund.dev/app/chart_data"

# ── Default download parameters ────────────────────────────────────
DEFAULT_RESOLUTION = "hour"
DEFAULT_REDUNDANCY_DAYS = 14        # Overlap for incremental updates
TSO_REDUNDANCY_HOURS = 72           # EMA's overlap for per-TSO updates
BOOTSTRAP_DAYS = 45                 # Days for bootstrapping new keys
```

**What's new vs EP/EMA:** EP had region mappings and filter keys but no TSO support. EMA had TSO support but no config module (everything hardcoded in the collection script). This merges both into one config, with national and TSO data sharing the same structure.

Note: The national filter keys (`SMARD_FILTER_KEYS` in `config/columns.py`, ~96 entries covering generation, forecasts, prices, flows) are already ported in stage 1. This module adds the API-specific config that `columns.py` doesn't cover.

---

### 2.3 Config: Commodity Constants

**`energy_forecasting/config/commodities.py`** (new file) — tickers, ICAP system IDs, date constants, price ranges. Ported from EP's `src/config/commodities.py`.

```python
"""Commodity data source configuration.

Constants for ICAP carbon allowances, Yahoo Finance commodities,
FRED gas price series, and Energy Charts fallback prices.
"""

from datetime import date

# ── Yahoo Finance tickers ───────────────────────────────────────────
TICKERS: dict[str, str] = {
    "ttf":             "TTF=F",     # TTF Natural Gas Futures (EUR/MWh)
    "brent":           "BZ=F",      # Brent Crude Oil Futures (USD/barrel)
    "carbon_realtime": "CO2.L",     # SparkChange EUA ETC (GBP, proxy)
}

# ── ICAP carbon system IDs ──────────────────────────────────────────
ICAP_SYSTEMS: dict[str, int] = {
    "eu_ets_phase3": 33,    # 2014-2018
    "eu_ets_phase4": 35,    # 2019-present
}

# ── Output column names ────────────────────────────────────────────
COLUMN_NAMES: dict[str, str] = {
    "carbon":           "carbon_eur_per_ton",
    "carbon_realtime":  "carbon_realtime_eur_per_ton",
    "ttf":              "ttf_eur_per_mwh",
    "brent":            "brent_usd_per_barrel",
}

# ── Data availability start dates ──────────────────────────────────
# Used to set download start for each source. Note: Brent (BZ=F) has
# Yahoo data from ~2007, NOT just 2021 as some earlier docs suggested.
DATA_START: dict[str, date] = {
    "carbon_icap":      date(2014, 11, 3),
    "carbon_realtime":  date(2021, 10, 18),   # CO2.L listing date
    "ttf_yahoo":        date(2017, 10, 23),   # First Yahoo TTF data
    "brent":            date(2015, 1, 1),     # Yahoo has data from ~2007; 2015 matches SMARD start
    "fred_eu_gas":      date(2002, 1, 1),     # Monthly, long history
    "fred_us_gas":      date(2002, 1, 1),
}

# ── Price range validation ──────────────────────────────────────────
# (min, max) — values outside these are flagged as suspect
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "carbon":  (3.0, 200.0),      # EUR/ton
    "ttf":     (5.0, 350.0),      # EUR/MWh (peaked ~339 Aug 2022)
    "brent":   (10.0, 200.0),     # USD/barrel
}

# ── FRED series identifiers ────────────────────────────────────────
FRED_SERIES: dict[str, str] = {
    "eu_gas_monthly": "PNGASEUUSDM",   # EU natural gas import price (USD/MMBtu)
    "us_gas_daily":   "DHHNGSP",        # US Henry Hub spot price (USD/MMBtu)
}

# ── TTF reconstruction constants ────────────────────────────────────
TTF_YAHOO_START = date(2017, 10, 23)        # First Yahoo TTF data
UNIT_CONVERSION_MMBTU_TO_MWH = 0.293        # 1 MMBtu = 0.293 MWh
TTF_RECONSTRUCTION_START = date(2014, 12, 1) # Start of gap period

# ── Energy Charts ───────────────────────────────────────────────────
ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info"

ENERGY_CHARTS_SERIES: dict[str, dict] = {
    "da_price_de_lu": {
        "bzn": "DE-LU",
        "column": "day_ahead_price_de_lu_eur_per_mwh",
    },
}
```

---

### 2.4 SMARD API Client

**`energy_forecasting/data/smard.py`** — low-level HTTP client for the SMARD API. Ported from EP's `src/data/smard.py`. Handles URL construction, timestamp fetching, data retrieval, and error handling.

```python
"""Low-level SMARD API client.

All functions return DataFrames or raise DataNotAvailableError.
No file I/O — that's handled by SmardSource in sources.py.
"""

import pandas as pd
import requests
from loguru import logger

from energy_forecasting.config.smard import SMARD_API_BASE


class DataNotAvailableError(Exception):
    """Raised when a SMARD filter/region combination returns 404."""


def get_timestamps(
    filter_id: int, region: str, resolution: str = "hour"
) -> list[int]:
    """Fetch available weekly-chunk timestamps for a filter/region.

    Returns list of timestamps in milliseconds. These are the valid
    chunk IDs for get_data() — roughly one per week.

    Raises DataNotAvailableError if the combination doesn't exist.
    """
    url = f"{SMARD_API_BASE}/{filter_id}/{region}/index_{resolution}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        raise DataNotAvailableError(
            f"No data for filter={filter_id}, region={region}"
        )
    resp.raise_for_status()
    return resp.json()["timestamps"]


def get_data(
    filter_id: int, region: str, timestamp: int, resolution: str = "hour"
) -> pd.DataFrame:
    """Fetch one weekly chunk of data.

    Returns DataFrame with columns: timestamp (ms), value, time (UTC).
    """
    url = (
        f"{SMARD_API_BASE}/{filter_id}/{region}/"
        f"{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    resp = requests.get(url, timeout=30)
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
) -> pd.DataFrame:
    """Fetch all data for a filter/region, optionally from specific timestamps.

    If timestamp_list is None, fetches all available timestamps.
    Returns concatenated DataFrame with UTC DatetimeIndex.
    """
    if timestamp_list is None:
        timestamp_list = get_timestamps(filter_id, region, resolution)

    if not timestamp_list:
        return pd.DataFrame()

    chunks = []
    for ts in timestamp_list:
        chunk = get_data(filter_id, region, ts, resolution)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["value"])       # Drop unfilled future rows
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("time").sort_index()
    return df[["value"]]
```

**Key differences from EP:**
- Returns DataFrames with UTC DatetimeIndex (EP returned raw columns with millisecond timestamps).
- Drops NaN values and deduplicates inline (EP did this in SmardSource).
- No `region` or `measure` columns in output (unnecessary metadata — SmardSource knows what it asked for).
- Uses `loguru` instead of `print`.

---

### 2.5 DataSource Base Class

**`energy_forecasting/data/sources.py`** — abstract base for all data sources. Ported from EP's DataSource pattern, enhanced with EMA's crash-resilient saves.

```python
"""DataSource base class and all source implementations.

Each source manages a Parquet file (or directory of Parquet files).
download() fetches full history from scratch.
update() incrementally appends new data with a redundancy window.

Design:
- All outputs are Parquet (no CSV intermediate).
- Each source writes via data/io.py (zstd compression, dtype reduction).
- Sources own their output path and update logic.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

from energy_forecasting.data.io import load_parquet, save_parquet


class DataSource(ABC):
    """Base class for incremental data sources.

    Subclasses implement fetch_all() for full history and
    fetch_update() for incremental data. The base class handles
    the merge-with-existing logic.
    """

    @property
    @abstractmethod
    def output_path(self) -> Path:
        """Path to this source's Parquet file."""

    @abstractmethod
    def fetch_all(self) -> pd.DataFrame:
        """Fetch full history. Returns DataFrame with DatetimeIndex (UTC)."""

    @abstractmethod
    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Fetch data from last_timestamp onwards (with redundancy overlap).

        Returns DataFrame with DatetimeIndex (UTC). May overlap with
        existing data — the merge logic handles deduplication.
        """

    def download(self) -> None:
        """Full download from scratch. Overwrites existing data."""
        logger.info(f"Downloading full history: {self.output_path.name}")
        df = self.fetch_all()
        if df.empty:
            logger.warning(f"No data returned for {self.output_path.name}")
            return
        save_parquet(df, self.output_path)
        logger.info(
            f"Saved {self.output_path.name}: "
            f"{len(df)} rows, {df.index.min()} to {df.index.max()}"
        )

    def update(self) -> None:
        """Incremental update. Fetches new data and merges with existing."""
        if not self.output_path.exists():
            logger.info(f"No existing data, running full download")
            self.download()
            return

        existing = load_parquet(self.output_path)
        last_ts = existing.index.max()
        logger.info(f"Updating {self.output_path.name} from {last_ts}")

        new = self.fetch_update(last_ts)
        if new.empty:
            logger.info("No new data")
            return

        # Merge: keep new data where timestamps overlap
        merged = pd.concat([existing, new])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        save_parquet(merged, self.output_path)
        new_rows = len(merged) - len(existing)
        logger.info(
            f"Updated {self.output_path.name}: "
            f"+{new_rows} rows, now {merged.index.min()} to {merged.index.max()}"
        )
```

**What's new vs EP:**
- Parquet-native (EP used CSV with `pd.read_csv` and `to_csv`).
- `output_path` is a property (EP passed paths as arguments).
- Base class handles the merge-and-deduplicate pattern (EP's base `update()` did this but SmardSource overrode it entirely).
- Subclasses only implement `fetch_all()` and `fetch_update()` — cleaner separation of fetch vs storage.

**Design note:** `SmardSource` and `OpenMeteoSource` override `download()`/`update()` because they manage multiple columns or files with different logic. Simple sources (Yahoo, FRED) use the base class directly.

---

### 2.6 SmardSource

Handles both national (DE-LU, DE-AT-LU) and per-TSO (50Hertz, Amprion, etc.) SMARD data. One Parquet file per region, columns added/updated independently.

```python
class SmardSource(DataSource):
    """SMARD data for a single region (national or TSO).

    National regions (DE_LU, DE_AT_LU): generation, load, prices,
    forecasts, cross-border flows — all filter keys for that region.

    TSO regions (50Hertz, Amprion, etc.): generation and load only,
    using the per-TSO filter keys from config/smard.py.

    Each region produces one Parquet file with columns named by
    clean_column_name() for national, or by TSO_FILTER_KEYS values
    with TSO suffix for per-TSO.
    """

    def __init__(self, region: str, resolution: str = "hour"):
        self.region = region
        self.resolution = resolution
        self._is_tso = region in TSO_REGIONS

    @property
    def output_path(self) -> Path:
        if self._is_tso:
            return SMARD_DIR / "tso" / f"{self.region}.parquet"
        return SMARD_DIR / f"{self.region}.parquet"

    @property
    def filter_keys(self) -> dict[int, str]:
        """Filter keys valid for this region.

        National: SMARD_FILTER_KEYS + region-specific cross-border flows,
        minus excluded keys (installed capacity, scheduled commercial).
        TSO: TSO_FILTER_KEYS minus known-missing combinations.
        """
        if self._is_tso:
            return {
                k: v for k, v in TSO_FILTER_KEYS.items()
                if (k, self.region) not in KNOWN_MISSING
            }
        # National region: combine SMARD national keys with cross-border flows
        flow_dict = (
            CROSS_BORDER_DE_LU if self.region == "DE-LU"
            else CROSS_BORDER_DE_AT_LU
        )
        combined = {
            k: v for k, v in SMARD_FILTER_KEYS.items()
            if k not in EXCLUDED_KEYS
        }
        combined.update(flow_dict)
        return combined

    def _column_name(self, filter_id: int, base_name: str) -> str:
        """Column name for a given filter in this region."""
        if self._is_tso:
            return f"{base_name}{TSO_SUFFIXES[self.region]}"
        return SMARD_COLUMN_NAMES[filter_id]

    def download(self) -> None:
        """Download all filter keys for this region.

        Uses ThreadPoolExecutor for parallel API calls, then
        merges columns sequentially into a single Parquet file.
        Crash-resilient: existing columns in the output file are
        skipped on resume (EMA pattern).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            f"Downloading SMARD {self.region} "
            f"({len(self.filter_keys)} filter keys)"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing columns (resume support)
        existing_cols = set()
        if self.output_path.exists():
            existing_cols = set(load_parquet(self.output_path).columns)

        def fetch_one(filter_id: int, name: str) -> tuple[str, pd.DataFrame]:
            col = self._column_name(filter_id, name)
            if col in existing_cols:
                logger.debug(f"Skipping {col} (already exists)")
                return col, pd.DataFrame()
            try:
                df = get_all_data(filter_id, self.region, self.resolution)
                df = df.rename(columns={"value": col})
                return col, df
            except DataNotAvailableError:
                logger.warning(f"No data: {filter_id}/{self.region}")
                return col, pd.DataFrame()

        results = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(fetch_one, fid, name): (fid, name)
                for fid, name in self.filter_keys.items()
            }
            for future in as_completed(futures):
                col, df = future.result()
                if not df.empty:
                    results[col] = df[col]

        if not results:
            logger.warning("No data fetched")
            return

        # Build or extend the Parquet file
        if self.output_path.exists():
            combined = load_parquet(self.output_path)
            for col, series in results.items():
                combined = _merge_column(combined, col, series)
        else:
            combined = pd.DataFrame(results)

        combined = combined.sort_index()
        save_parquet(combined, self.output_path)
        logger.info(
            f"Saved {self.output_path.name}: {len(combined)} rows, "
            f"{len(combined.columns)} columns"
        )

    def update(self) -> None:
        """Incremental update with redundancy window.

        SMARD data comes in weekly chunks identified by millisecond timestamps.
        You can't request "data from date X" — you must fetch the timestamp
        index first, then download the right chunks. EP's bisect-based approach:

        1. Fetch timestamp index from API (one call per filter key)
        2. Convert local cutoff (last_ts - redundancy) to milliseconds
        3. bisect_left to find starting chunk index
        4. Download chunks from that index forward (parallel)
        5. Merge into existing Parquet (per-column, keep-new on overlap)

        The redundancy window (default 14 days) re-fetches recent chunks
        because SMARD retroactively corrects data for several days after
        initial publication.

        New columns (not yet in the Parquet file) are bootstrapped with
        a full download limited to the last BOOTSTRAP_DAYS days.
        """
        import bisect
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self.output_path.exists():
            self.download()
            return

        existing = load_parquet(self.output_path)
        existing_cols = set(existing.columns)

        # Identify new columns to bootstrap
        all_filter_items = list(self.filter_keys.items())
        missing_items = [
            (fid, name) for fid, name in all_filter_items
            if self._column_name(fid, name) not in existing_cols
        ]
        update_items = [
            (fid, name) for fid, name in all_filter_items
            if self._column_name(fid, name) in existing_cols
        ]

        # Bootstrap missing columns (last BOOTSTRAP_DAYS days only)
        if missing_items:
            logger.info(
                f"Bootstrapping {len(missing_items)} new columns "
                f"from last {BOOTSTRAP_DAYS} days"
            )
            bootstrap_cutoff_ms = int(
                (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=BOOTSTRAP_DAYS))
                .timestamp() * 1000
            )
            for fid, name in missing_items:
                try:
                    all_ts = get_timestamps(fid, self.region, self.resolution)
                    start_idx = bisect.bisect_left(all_ts, bootstrap_cutoff_ms)
                    df = get_all_data(
                        fid, self.region, self.resolution,
                        timestamp_list=all_ts[start_idx:],
                    )
                    if not df.empty:
                        col = self._column_name(fid, name)
                        existing = _merge_column(
                            existing, col, df["value"].rename(col)
                        )
                except DataNotAvailableError:
                    pass

        # Incremental update for existing columns
        last_ts = existing.index.max()
        cutoff_ms = int(
            (last_ts - pd.Timedelta(days=DEFAULT_REDUNDANCY_DAYS))
            .timestamp() * 1000
        )

        def fetch_update_one(fid: int, name: str) -> tuple[str, pd.Series]:
            col = self._column_name(fid, name)
            try:
                all_ts = get_timestamps(fid, self.region, self.resolution)
                start_idx = bisect.bisect_left(all_ts, cutoff_ms)
                df = get_all_data(
                    fid, self.region, self.resolution,
                    timestamp_list=all_ts[start_idx:],
                )
                if not df.empty:
                    return col, df["value"].rename(col)
            except DataNotAvailableError:
                pass
            return col, pd.Series(dtype=float)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(fetch_update_one, fid, name): (fid, name)
                for fid, name in update_items
            }
            for future in as_completed(futures):
                col, series = future.result()
                if not series.empty:
                    existing = _merge_column(existing, col, series)

        save_parquet(existing.sort_index(), self.output_path)
        logger.info(
            f"Updated {self.output_path.name}: "
            f"{len(existing)} rows, {len(existing.columns)} columns"
        )

    # fetch_all / fetch_update are not used directly (download/update overridden)
    # but provided for interface compliance
    def fetch_all(self) -> pd.DataFrame:
        raise NotImplementedError("Use download() directly")

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError("Use update() directly")


def _merge_column(
    df: pd.DataFrame, col_name: str, series: pd.Series
) -> pd.DataFrame:
    """Merge a single column into an existing DataFrame.

    New timestamps extend the index. Overlapping timestamps are
    overwritten (keep new).
    """
    if col_name in df.columns:
        # Update existing column
        combined_index = df.index.union(series.index)
        df = df.reindex(combined_index)
        df.loc[series.index, col_name] = series.values
    else:
        # Add new column
        combined_index = df.index.union(series.index)
        df = df.reindex(combined_index)
        df[col_name] = series.reindex(combined_index)
    return df
```

**Key design decisions:**
- **One class for national + TSO.** The SMARD API is the same; only the region code and filter keys differ. `_is_tso` flag controls the column naming convention.
- **Column-level granularity.** Each filter key becomes a column. Crash resilience: if download fails after 30/50 keys, existing columns are preserved and skipped on resume (EMA pattern).
- **Parallel fetch, sequential merge.** ThreadPoolExecutor for API calls (the bottleneck), sequential DataFrame operations (fast).
- **No CSV intermediate.** EP wrote CSVs per key then combined to Parquet. We go directly to one Parquet per region.

**Column naming:**
- National: `clean_column_name()` from `config/columns.py` (e.g., `stromerzeugung_wind_onshore`)
- Per-TSO: `{base_name}{tso_suffix}` (e.g., `wind_onshore_50hz`, `solar_ampr`)

---

### 2.7 Open-Meteo Weather Collection

**`energy_forecasting/data/weather.py`** — three-endpoint weather data collection. Ported from EMA's `collect_data_openmeteo.py`. This is the most complex source because it manages three temporal scopes (actual, historical forecast, current forecast) across multiple locations.

```python
"""Open-Meteo weather data collection.

Three endpoints with different temporal coverage:
1. Archive API — historical actuals (2015+, hourly only)
2. Historical Forecast API — forecasts as issued (~2022+, hourly + 15min)
3. Forecast API — current 14-day forecast (hourly + 15min)

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

from energy_forecasting.config import CACHE_DIR, LOCATIONS_DIR, WEATHER_DIR
from energy_forecasting.data.io import load_parquet, save_parquet

# ── Variable sets ───────────────────────────────────────────────────
# Ported from EMA's collect_data_openmeteo.py

VARS_BASIC: list[str] = [
    "temperature_2m",           # degC
    "relative_humidity_2m",     # %
    "surface_pressure",         # hPa
    "precipitation",            # mm
    "cloud_cover",              # %
]

VARS_WIND: list[str] = [
    "wind_speed_10m",           # km/h
    "wind_speed_100m",          # km/h
    "wind_direction_10m",       # deg
    "wind_direction_100m",      # deg
    "wind_gusts_10m",           # km/h
]

VARS_RADIATION: list[str] = [
    "shortwave_radiation",      # W/m2
    "direct_radiation",         # W/m2
    "diffuse_radiation",        # W/m2
    "direct_normal_irradiance", # W/m2
    "global_tilted_irradiance", # W/m2
    "terrestrial_radiation",    # W/m2
]

# Which variable groups each asset type uses
ASSET_VARIABLES: dict[str, list[str]] = {
    "offshore": VARS_BASIC + VARS_WIND,
    "onshore":  VARS_BASIC + VARS_WIND,
    "solar":    VARS_BASIC + VARS_RADIATION,
    "cities":   VARS_BASIC + VARS_WIND + VARS_RADIATION,
}

# ── Physical limits ─────────────────────────────────────────────────
# Values outside these bounds are clipped. Ported from EMA.
PHYSICAL_LIMITS: dict[str, tuple[float, float]] = {
    "temperature_2m":           (-45, 50),
    "relative_humidity_2m":     (0, 100),
    "surface_pressure":         (900, 1080),
    "precipitation":            (0, 100),
    "cloud_cover":              (0, 100),
    "wind_speed_10m":           (0, 200),
    "wind_speed_100m":          (0, 200),
    "wind_direction_10m":       (0, 360),
    "wind_direction_100m":      (0, 360),
    "wind_gusts_10m":           (0, 300),
    "shortwave_radiation":      (0, 1400),
    "direct_radiation":         (0, 1200),
    "diffuse_radiation":        (0, 450),
    "direct_normal_irradiance": (0, 1200),
    "global_tilted_irradiance": (0, 1400),
    "terrestrial_radiation":    (200, 2000),
}

# ── API URLs ────────────────────────────────────────────────────────
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def load_locations(asset_type: str, tso: str) -> list[dict]:
    """Load locations for a given asset type and TSO from eu_locations.json."""
    path = LOCATIONS_DIR / "eu_locations.json"
    with open(path) as f:
        data = json.load(f)

    # Map asset_type to location key in eu_locations.json
    type_key = {
        "offshore": "offshore",
        "onshore": "onshore",
        "solar": "solar",
        "cities": "cities",
    }[asset_type]

    for country in data["countries_metadata"]:
        if country["code"] == "DE":
            return [
                loc for loc in country["locations"].get(type_key, [])
                if loc["TSO"] == tso
            ]
    return []


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

        if not self.locations:
            logger.warning(f"No locations for {asset_type}/{tso}")

    def _path(self, scope: str) -> Path:
        return self.output_dir / f"{scope}.parquet"

    def download(self) -> None:
        """Full download of all three endpoints."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.locations:
            return

        # 1. Historical actuals (hourly only)
        logger.info(
            f"Fetching archive weather: {self.asset_type}/{self.tso} "
            f"({len(self.locations)} locations)"
        )
        df_actual = self._fetch_archive()
        if not df_actual.empty:
            if self.asset_type == "solar":
                df_actual = self._add_solar_columns(df_actual)
            save_parquet(df_actual, self._path("history"))

        # 2. Historical forecasts
        logger.info(f"Fetching historical forecasts: {self.asset_type}/{self.tso}")
        df_hist_fc = self._fetch_historical_forecast()
        if not df_hist_fc.empty:
            if self.asset_type == "solar":
                df_hist_fc = self._add_solar_columns(df_hist_fc)
            save_parquet(df_hist_fc, self._path("hist_forecast"))

        # 3. Current forecast
        logger.info(f"Fetching current forecast: {self.asset_type}/{self.tso}")
        df_forecast = self._fetch_current_forecast()
        if not df_forecast.empty:
            if self.asset_type == "solar":
                df_forecast = self._add_solar_columns(df_forecast)
            save_parquet(df_forecast, self._path("forecast"))

    def update(self) -> None:
        """Incremental update with 3-day overlap.

        - Archive: extend from (last_date - 3 days) to yesterday
        - Historical forecast: extend from (last_date - 3 days) to yesterday
        - Current forecast: replace entirely (it's the latest 14-day window)
        - Actual data overwrites forecast data where they overlap
        """
        if not self._path("history").exists():
            self.download()
            return

        if not self.locations:
            return

        overlap_days = 3

        # Update archive (actual weather)
        existing = load_parquet(self._path("history"))
        update_start = (
            existing.index.max() - pd.Timedelta(days=overlap_days)
        ).strftime("%Y-%m-%d")
        df_new = self._fetch_archive(start_override=update_start)
        if not df_new.empty:
            if self.asset_type == "solar":
                df_new = self._add_solar_columns(df_new)
            merged = pd.concat([existing, df_new])
            merged = merged[~merged.index.duplicated(keep="last")]
            save_parquet(merged.sort_index(), self._path("history"))

        # Update historical forecast (same overlap pattern)
        # ...

        # Replace current forecast
        df_forecast = self._fetch_current_forecast()
        if not df_forecast.empty:
            if self.asset_type == "solar":
                df_forecast = self._add_solar_columns(df_forecast)
            save_parquet(df_forecast, self._path("forecast"))

    def _fetch_archive(self, start_override: str | None = None) -> pd.DataFrame:
        """Fetch from Open-Meteo Archive API.

        One request per location. Each request returns all variables for
        that location. Column names get the location suffix appended.
        """
        import openmeteo_requests
        from requests_cache import CachedSession
        from retry_requests import retry

        cache = CachedSession(str(CACHE_DIR / "openmeteo"), expire_after=-1)
        session = retry(cache, retries=5, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=session)

        start = start_override or self.start_date
        end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        all_dfs = []
        for loc in self.locations:
            params = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "start_date": start,
                "end_date": end,
                "hourly": self.variables,
                "timezone": "UTC",
            }
            responses = client.weather_api(ARCHIVE_URL, params=params)
            hourly = responses[0].Hourly()

            # Build DataFrame from response
            time_range = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
            data = {}
            for i, var in enumerate(self.variables):
                col = f"{var}{loc['suffix']}"
                values = hourly.Variables(i).ValuesAsNumpy()
                data[col] = self._validate_physical(var, values)

            loc_df = pd.DataFrame(data, index=time_range)
            all_dfs.append(loc_df)

        if not all_dfs:
            return pd.DataFrame()

        # Join all locations (outer join preserves all timestamps)
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.join(df, how="outer")

        return result

    def _fetch_historical_forecast(self, ...) -> pd.DataFrame:
        """Same pattern as _fetch_archive but using HIST_FORECAST_URL."""
        ...

    def _fetch_current_forecast(self) -> pd.DataFrame:
        """Same pattern but using FORECAST_URL with forecast_days=14."""
        ...

    def _validate_physical(
        self, variable: str, values: np.ndarray
    ) -> np.ndarray:
        """Clip values to physical limits."""
        if variable in PHYSICAL_LIMITS:
            lo, hi = PHYSICAL_LIMITS[variable]
            return np.clip(values, lo, hi)
        return values

    def _add_solar_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute solar elevation and azimuth for each location.

        Ported from EMA's add_solar_elevation_and_azimuth().
        Uses pysolar — computed at collection time so the columns
        are available for feature engineering without re-computation.
        """
        for loc in self.locations:
            suffix = loc["suffix"]
            elevations = []
            azimuths = []
            for ts in df.index:
                dt = ts.to_pydatetime()
                elev = get_altitude(loc["lat"], loc["lon"], dt)
                az = get_azimuth(loc["lat"], loc["lon"], dt)
                elevations.append(elev)
                azimuths.append(az)
            df[f"solar_elevation_deg{suffix}"] = elevations
            df[f"solar_azimuth_deg{suffix}"] = azimuths
        return df
```

**Key design decisions:**

- **Three files per (type, TSO)** — matches EMA's pattern. Actual weather and forecast weather are kept separate because they serve different purposes: actuals for training, forecasts for inference.
- **Solar elevation at collection time, solar asset type only** — ported from EMA. Computing it here (rather than in feature engineering) avoids re-computing for every feature set iteration. Only applied to solar locations (matching EMA's conditional `if 'solar' in datadir`); wind and city locations don't need sun position.
- **Physical limit validation during collection** — clip out-of-range values immediately rather than discovering them downstream.
- **Location-suffixed columns** — each variable gets the location suffix (e.g., `temperature_2m_city_berlin`, `wind_speed_100m_won_hueselitz`). This enables per-location spatial aggregation in stage 4.
- **`openmeteo_requests` client with retry** — EMA's approach. The API has rate limits; the retry/cache stack handles transient failures.

**Important EMA gotcha to carry forward:** 15-minute data uses `wind_speed_80m` instead of `wind_speed_100m`. If 15-minute support is added later, the variable lists need adjustment.

---

### 2.8 Commodity Sources & Reconstruction

**`energy_forecasting/data/commodities.py`** — download functions for each commodity source, plus TTF gap reconstruction and carbon dual-source merging. Ported from EP's `src/data/commodities.py` (non-trivial, ~300 lines of reconstruction logic).

```python
"""Commodity data collection and reconstruction.

Sources:
- ICAP: EU carbon allowances (Phase 3: 2014-2018, Phase 4: 2019+)
- Yahoo Finance: TTF gas, Brent oil, CO2.L realtime carbon
- FRED: EU/US gas prices for TTF gap reconstruction

Reconstruction:
- TTF: Yahoo data starts Oct 2017. The Dec 2014-Oct 2017 gap is
  filled using FRED EU/US gas prices with bias correction.
- Carbon: ICAP (official, ~60-day lag) merged with CO2.L (realtime)
  using bias correction in the overlap period.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred
from loguru import logger

from energy_forecasting.config import COMMODITIES_DIR
from energy_forecasting.config.commodities import (
    COLUMN_NAMES,
    DATA_START,
    FRED_SERIES,
    ICAP_SYSTEMS,
    PRICE_RANGES,
    TICKERS,
    TTF_RECONSTRUCTION_START,
    TTF_YAHOO_START,
    UNIT_CONVERSION_MMBTU_TO_MWH,
)
from energy_forecasting.data.io import load_parquet, save_parquet


# ── Individual source downloads ─────────────────────────────────────

ICAP_URL = "https://allowancepriceexplorer.icapcarbonaction.com/systems/reports/price/download"

def _download_icap() -> pd.DataFrame:
    """Internal: fetch ICAP data and return as DataFrame.

    Used by IcapSource.fetch_all() and fetch_update().
    Separated from the class so the ICAP URL/param logic is in one place.
    """
    import requests
    from io import StringIO

    frames = []
    for phase_name, system_id in ICAP_SYSTEMS.items():
        start_date = DATA_START["carbon_icap"]
        params = {
            "systemIds": system_id,
            "startDate": int(
                pd.Timestamp(start_date, tz="UTC").timestamp() * 1000
            ),
            "endDate": int(
                pd.Timestamp.now(tz="UTC").timestamp() * 1000
            ),
        }
        resp = requests.get(ICAP_URL, params=params, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # ... parse dates, rename columns
        frames.append(df)

    combined = pd.concat(frames).drop_duplicates().sort_index()
    combined.columns = ["carbon_primary", "carbon_secondary", "eur_usd_rate"]
    return combined


# ── Reconstruction ──────────────────────────────────────────────────

def reconstruct_ttf(raw_dir: Path) -> pd.Series:
    """Reconstruct TTF price series with FRED gap-filling.

    Yahoo TTF starts Oct 2017. Gap from Dec 2014 to Oct 2017 is
    filled using:
    1. FRED EU monthly gas (USD/MMBtu) -> EUR/MWh via ICAP EUR/USD rate
    2. FRED US daily gas (USD/MMBtu) -> EUR/MWh, centered (removes level)
    3. Reconstructed = EU baseline + US daily variation
    4. Bias-corrected against Yahoo TTF in the overlap period

    Returns: Series with DatetimeIndex, name='ttf_eur_per_mwh'.
    Ported from EP's src/data/commodities.py (~line 548-679).
    """
    # Load raw sources
    ttf_yahoo = load_parquet(raw_dir / "ttf.parquet")["price"]
    fred_eu = load_parquet(raw_dir / "fred_eu_gas.parquet")["price"]
    fred_us = load_parquet(raw_dir / "fred_us_gas.parquet")["price"]
    icap = load_parquet(raw_dir / "icap.parquet")

    # Step 1: Unit conversion (USD/MMBtu -> EUR/MWh)
    eur_usd = icap["eur_usd_rate"].reindex(fred_eu.index, method="ffill")
    eu_gas_eur = (fred_eu / eur_usd) * UNIT_CONVERSION_MMBTU_TO_MWH
    eu_gas_daily = eu_gas_eur.resample("D").ffill()

    eur_usd_daily = icap["eur_usd_rate"].reindex(
        fred_us.index, method="ffill"
    )
    us_gas_eur = (fred_us / eur_usd_daily) * UNIT_CONVERSION_MMBTU_TO_MWH

    # Step 2: Bias correction (EU adjusted to match Yahoo in overlap)
    overlap_start = pd.Timestamp(TTF_YAHOO_START, tz="UTC")
    eu_overlap = eu_gas_daily.loc[overlap_start:]
    yahoo_overlap = ttf_yahoo.reindex(eu_overlap.index)
    valid = eu_overlap.notna() & yahoo_overlap.notna()
    if valid.any():
        bias = (yahoo_overlap[valid] - eu_overlap[valid]).mean()
        corr = yahoo_overlap[valid].corr(eu_overlap[valid])
        logger.info(
            f"TTF bias correction: {bias:.2f} EUR/MWh, r={corr:.3f}"
        )
    else:
        bias = 0.0
    eu_adjusted = eu_gas_daily + bias

    # Step 3: US daily variation (centered — removes level, keeps shape)
    us_monthly_mean = us_gas_eur.resample("MS").mean()
    us_centered = us_gas_eur - us_monthly_mean.reindex(
        us_gas_eur.index, method="ffill"
    )

    # Step 4: Combine for gap period
    gap_start = pd.Timestamp(TTF_RECONSTRUCTION_START, tz="UTC")
    gap_end = overlap_start - pd.Timedelta(days=1)

    # Baseline (EU adjusted, forward-filled to daily) + daily variation (US)
    combined_idx = eu_adjusted.loc[gap_start:gap_end].index.union(
        us_centered.loc[gap_start:gap_end].index
    )
    baseline = eu_adjusted.reindex(combined_idx, method="ffill")
    variation = us_centered.reindex(combined_idx).fillna(0)
    reconstructed = baseline + variation

    # Combine: reconstructed gap + Yahoo actual
    full = pd.concat([reconstructed, ttf_yahoo])
    full = full[~full.index.duplicated(keep="last")].sort_index()
    full.name = COLUMN_NAMES["ttf"]

    # Validate
    lo, hi = PRICE_RANGES["ttf"]
    outliers = (full < lo) | (full > hi)
    if outliers.any():
        logger.warning(
            f"TTF: {outliers.sum()} values outside [{lo}, {hi}]"
        )

    return full


def merge_carbon(raw_dir: Path) -> pd.DataFrame:
    """Merge ICAP historical carbon with CO2.L realtime.

    ICAP is official but has ~60-day publication lag.
    CO2.L (SparkChange EUA ETC) is realtime but only from Oct 2021.
    Bias-correct CO2.L to match ICAP in the overlap period, then
    use CO2.L to extend forward where ICAP is missing.

    Returns: DataFrame with columns [carbon_eur_per_ton,
    carbon_realtime_eur_per_ton] and daily DatetimeIndex.
    Ported from EP's src/data/commodities.py (~line 682-820).
    """
    icap = load_parquet(raw_dir / "icap.parquet")
    co2l = load_parquet(raw_dir / "carbon_realtime.parquet")

    carbon_col = COLUMN_NAMES["carbon"]
    carbon_rt_col = COLUMN_NAMES["carbon_realtime"]

    # ICAP: combine phase 3 + phase 4, rename
    carbon = icap["carbon_primary"].rename(carbon_col)
    co2l_price = co2l["price"].rename(carbon_rt_col)

    # Merge on date
    combined = pd.concat([carbon, co2l_price], axis=1)

    # Bias correction in overlap period
    overlap = combined.dropna()
    if len(overlap) > 0:
        bias = (overlap[carbon_col] - overlap[carbon_rt_col]).mean()
        corr = overlap[carbon_col].corr(overlap[carbon_rt_col])
        logger.info(
            f"Carbon bias: {bias:.2f} EUR/ton, r={corr:.3f}"
        )
    else:
        bias = 0.0

    # Fill ICAP gaps with bias-corrected CO2.L
    missing_icap = combined[carbon_col].isna()
    combined.loc[missing_icap, carbon_col] = (
        combined.loc[missing_icap, carbon_rt_col] + bias
    )

    # Forward-fill within valid data range only (EP's bounded ffill)
    for col in combined.columns:
        first_valid = combined[col].first_valid_index()
        last_valid = combined[col].last_valid_index()
        if first_valid is not None:
            mask = (combined.index >= first_valid) & (
                combined.index <= last_valid
            )
            combined.loc[mask, col] = combined.loc[mask, col].ffill()

    return combined


class IcapSource(DataSource):
    """ICAP carbon allowance prices (Phase 3 + Phase 4)."""

    @property
    def output_path(self) -> Path:
        return COMMODITIES_DIR / "icap.parquet"

    def fetch_all(self) -> pd.DataFrame:
        return _download_icap()

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        # ICAP data is small (daily, ~3000 rows). Re-download full history
        # is simpler and more reliable than incremental.
        return _download_icap()


class YahooSource(DataSource):
    """Single Yahoo Finance ticker (TTF, Brent, or CO2.L)."""

    def __init__(self, ticker_key: str):
        self.ticker_key = ticker_key

    @property
    def output_path(self) -> Path:
        return COMMODITIES_DIR / f"{self.ticker_key}.parquet"

    def fetch_all(self) -> pd.DataFrame:
        import yfinance as yf

        ticker = TICKERS[self.ticker_key]
        start = DATA_START.get(
            f"{self.ticker_key}_yahoo", DATA_START.get(self.ticker_key)
        )
        df = yf.download(ticker, start=str(start), progress=False)
        df = df[["Close", "Volume"]].rename(
            columns={"Close": "price", "Volume": "volume"}
        )
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        import yfinance as yf

        ticker = TICKERS[self.ticker_key]
        start = (last_timestamp - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, progress=False)
        df = df[["Close", "Volume"]].rename(
            columns={"Close": "price", "Volume": "volume"}
        )
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df


class FredSource(DataSource):
    """FRED gas price series. Requires FRED_API_KEY env var."""

    def __init__(self, series_key: str):
        self.series_key = series_key

    @property
    def output_path(self) -> Path:
        return COMMODITIES_DIR / f"{self.series_key}.parquet"

    def fetch_all(self) -> pd.DataFrame:
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            logger.warning("FRED_API_KEY not set, skipping")
            return pd.DataFrame()

        fred = Fred(api_key=api_key)
        series = fred.get_series(FRED_SERIES[self.series_key])
        df = series.to_frame(name="price")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        # FRED data is small; re-fetch full history
        return self.fetch_all()


# ── All commodity sources (for CLI iteration) ─────────────────────

def all_commodity_sources() -> list[DataSource]:
    """Return all commodity DataSource instances."""
    return [
        IcapSource(),
        YahooSource("ttf"),
        YahooSource("brent"),
        YahooSource("carbon_realtime"),
        FredSource("fred_eu_gas"),
        FredSource("fred_us_gas"),
    ]
```

**Key design decisions:**

- **DataSource subclasses for raw downloads, standalone functions for reconstruction.** Raw commodity downloads (ICAP, Yahoo, FRED) fit the DataSource pattern cleanly — each has a Parquet file, a download path, and an update path. The reconstruction logic (TTF gap-fill, carbon merge) operates on already-downloaded raw files and belongs in stage 3's merge pipeline.
- **Reconstruction is computed, not cached.** The raw source files are the persistent state. `reconstruct_ttf()` and `merge_carbon()` are called during the stage 3 merge pipeline. The data is small (a few thousand daily rows), so recomputation is instant.
- **Bias correction carried forward from EP.** Both TTF and carbon use the same pattern: find overlap period, compute mean difference, apply offset. The correlation coefficient is logged for monitoring.
- **FRED_API_KEY required.** The FRED download needs an API key (free registration). If missing, the FRED download is skipped with a warning — TTF reconstruction will fail later unless the raw FRED files already exist.
- **Bounded forward-fill.** EP's pattern: only forward-fill within `[first_valid, last_valid]` — never extrapolate before data starts or after it ends. This avoids filling weekends/holidays beyond the last known data point.

---

### 2.9 Energy Charts Source

**Part of `energy_forecasting/data/sources.py`** — fallback day-ahead price source. Simple, uses the base DataSource class.

```python
class EnergyChartsSource(DataSource):
    """Day-ahead prices from energy-charts.info.

    Used as a fallback when SMARD has gaps. Not critical for
    normal operation.
    """

    def __init__(self, series_name: str = "da_price_de_lu"):
        self.config = ENERGY_CHARTS_SERIES[series_name]
        self._name = series_name

    @property
    def output_path(self) -> Path:
        return ENERGY_CHARTS_DIR / f"{self._name}.parquet"

    def fetch_all(self) -> pd.DataFrame:
        """Fetch full history from Energy Charts API."""
        bzn = self.config["bzn"]
        col = self.config["column"]
        # Port EP's fetch_price(bzn, start, end) logic
        ...

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Fetch from last_timestamp to tomorrow.

        Day-ahead prices for tomorrow are available from ~13:00 CET,
        so fetching through tomorrow+2 ensures full coverage.
        """
        ...
```

Low priority — this is only used when SMARD prices have gaps.

---

### 2.10 CLI Commands

**`energy_forecasting/cli.py`** — Typer app with download and update commands. Ported from EP's `src/cli.py` and EMA's `update_database.py`.

```python
"""CLI entry point.

Usage:
    energy-forecasting download smard --region DE-LU
    energy-forecasting download smard-tso --tso 50Hertz
    energy-forecasting download weather --type offshore --tso TenneT
    energy-forecasting download weather --all
    energy-forecasting download commodities
    energy-forecasting update all           # incremental update all
    energy-forecasting update smard         # incremental update SMARD only
"""

import typer

app = typer.Typer(help="Energy Forecasting CLI")


# ── Download commands ───────────────────────────────────────────────

download_app = typer.Typer(help="Download data from scratch")
app.add_typer(download_app, name="download")


@download_app.command("smard")
def download_smard(
    region: str = typer.Option("DE-LU", help="National region (DE-LU or DE-AT-LU)"),
    resolution: str = typer.Option("hour", help="Data resolution"),
):
    """Download national SMARD data for a region."""
    from energy_forecasting.data.sources import SmardSource

    source = SmardSource(region, resolution)
    source.download()


@download_app.command("smard-tso")
def download_smard_tso(
    tso: str = typer.Option(..., help="TSO name (50Hertz, Amprion, TenneT, TransnetBW, Creos)"),
    resolution: str = typer.Option("hour", help="Data resolution"),
):
    """Download per-TSO SMARD generation/load data."""
    from energy_forecasting.data.sources import SmardSource

    source = SmardSource(tso, resolution)
    source.download()


@download_app.command("weather")
def download_weather(
    asset_type: str = typer.Option(None, help="offshore, onshore, solar, or cities"),
    tso: str = typer.Option(None, help="TSO name"),
    all: bool = typer.Option(False, "--all", help="Download all type x TSO combinations"),
):
    """Download Open-Meteo weather data."""
    from energy_forecasting.data.weather import OpenMeteoSource
    from energy_forecasting.config.smard import TSO_REGIONS

    if all:
        for at in ["offshore", "onshore", "solar", "cities"]:
            for t in TSO_REGIONS:
                source = OpenMeteoSource(at, t)
                if source.locations:
                    source.download()
    else:
        if not asset_type or not tso:
            raise typer.BadParameter("Provide --asset-type and --tso, or use --all")
        OpenMeteoSource(asset_type, tso).download()


@download_app.command("commodities")
def download_commodities():
    """Download all commodity sources (ICAP, Yahoo, FRED)."""
    from energy_forecasting.data.commodities import all_commodity_sources

    for source in all_commodity_sources():
        source.download()


@download_app.command("energy-charts")
def download_energy_charts():
    """Download Energy Charts day-ahead prices (fallback source)."""
    from energy_forecasting.data.sources import EnergyChartsSource

    EnergyChartsSource().download()


@download_app.command("all")
def download_all_sources():
    """Download everything from scratch. Takes a long time."""
    from energy_forecasting.data.commodities import all_commodity_sources
    from energy_forecasting.data.sources import EnergyChartsSource, SmardSource
    from energy_forecasting.data.weather import OpenMeteoSource
    from energy_forecasting.config.smard import NATIONAL_REGIONS, TSO_REGIONS

    # National SMARD
    for region in NATIONAL_REGIONS:
        SmardSource(region).download()

    # Per-TSO SMARD
    for tso in TSO_REGIONS:
        SmardSource(tso).download()

    # Weather (all asset types x all TSOs)
    for asset_type in ["offshore", "onshore", "solar", "cities"]:
        for tso in TSO_REGIONS:
            source = OpenMeteoSource(asset_type, tso)
            if source.locations:
                source.download()

    # Commodities
    for source in all_commodity_sources():
        source.download()

    # Energy Charts
    EnergyChartsSource().download()


# ── Update commands ─────────────────────────────────────────────────

update_app = typer.Typer(help="Incremental data update")
app.add_typer(update_app, name="update")


@update_app.command("all")
def update_all_sources():
    """Incremental update of all data sources."""
    from energy_forecasting.data.commodities import all_commodity_sources
    from energy_forecasting.data.sources import EnergyChartsSource, SmardSource
    from energy_forecasting.data.weather import OpenMeteoSource
    from energy_forecasting.config.smard import NATIONAL_REGIONS, TSO_REGIONS

    for region in NATIONAL_REGIONS:
        SmardSource(region).update()

    for tso in TSO_REGIONS:
        SmardSource(tso).update()

    for asset_type in ["offshore", "onshore", "solar", "cities"]:
        for tso in TSO_REGIONS:
            source = OpenMeteoSource(asset_type, tso)
            if source.locations:
                source.update()

    for source in all_commodity_sources():
        source.update()

    EnergyChartsSource().update()


@update_app.command("smard")
def update_smard():
    """Update national + per-TSO SMARD data."""
    ...

@update_app.command("weather")
def update_weather():
    """Update all weather data."""
    ...

@update_app.command("commodities")
def update_commodities():
    """Update commodity sources."""
    ...
```

**Design notes:**
- Lazy imports inside commands (avoid loading ML libraries for data-only operations).
- `download all` is the full-from-scratch path. `update all` is the daily incremental path.
- Individual commands for each source type enable targeted re-downloads when something fails.
- `pyproject.toml` entry point: `[project.scripts] energy-forecasting = "energy_forecasting.cli:app"`.

---

### 2.11 Makefile Targets

Uncomment and implement the stage 2 targets in the Makefile:

```makefile
# ── Data targets ────────────────────────────────────────────────────

data:  ## Download all data from scratch
	energy-forecasting download all

update:  ## Incremental update all sources
	energy-forecasting update all

data-smard:  ## Download SMARD only (national + per-TSO)
	energy-forecasting download smard --region DE-LU
	energy-forecasting download smard --region DE-AT-LU
	energy-forecasting download smard-tso --tso 50Hertz
	energy-forecasting download smard-tso --tso Amprion
	energy-forecasting download smard-tso --tso TenneT
	energy-forecasting download smard-tso --tso TransnetBW
	energy-forecasting download smard-tso --tso Creos

data-weather:  ## Download weather only (all types x TSOs)
	energy-forecasting download weather --all

data-commodities:  ## Download commodities only
	energy-forecasting download commodities
```

---

### 2.12 Storage Layout

All raw data lives in `data/raw/` (gitignored). No CSV intermediate — everything is Parquet with zstd compression and dtype reduction via `data/io.py`.

```
data/raw/
├── smard/
│   ├── DE_LU.parquet              # National: ~50 columns (gen, load, prices,
│   │                              #   forecasts, cross-border flows), hourly
│   ├── DE_AT_LU.parquet           # Historical pre-Oct-2018 national data
│   └── tso/
│       ├── 50Hertz.parquet        # Per-TSO: ~12 columns (gen types + load)
│       ├── Amprion.parquet        #   Column names: {type}_{tso_suffix}
│       ├── TenneT.parquet         #   e.g., wind_onshore_50hz, solar_ampr
│       ├── TransnetBW.parquet
│       └── Creos.parquet
├── weather/
│   ├── offshore/
│   │   ├── TenneT/               # 5 offshore farms
│   │   │   ├── history.parquet    # Archive actuals (2015+, hourly)
│   │   │   ├── hist_forecast.parquet  # Historical forecasts (~2022+)
│   │   │   └── forecast.parquet   # Current 14-day forecast
│   │   └── 50Hertz/              # 2 offshore farms (Wikinger, Arkona)
│   │       ├── history.parquet
│   │       ├── hist_forecast.parquet
│   │       └── forecast.parquet
│   ├── onshore/
│   │   ├── 50Hertz/  (same 3 files)
│   │   ├── Amprion/
│   │   ├── TenneT/
│   │   └── TransnetBW/
│   ├── solar/
│   │   ├── 50Hertz/
│   │   ├── Amprion/
│   │   ├── TenneT/
│   │   └── TransnetBW/
│   └── cities/
│       ├── 50Hertz/
│       ├── Amprion/
│       ├── TenneT/
│       └── TransnetBW/
├── commodities/
│   ├── icap.parquet               # ICAP carbon (Phase 3 + 4 combined)
│   ├── ttf.parquet                # Yahoo TTF
│   ├── brent.parquet              # Yahoo Brent
│   ├── carbon_realtime.parquet    # Yahoo CO2.L
│   ├── fred_eu_gas.parquet        # FRED EU monthly gas
│   └── fred_us_gas.parquet        # FRED US daily gas
└── energy_charts/
    └── da_price_de_lu.parquet     # Fallback day-ahead prices
```

**Notes:**
- Creos is excluded from weather (Luxembourg has negligible generation in our scope; EMA doesn't collect weather for it).
- Offshore weather for TenneT (5 farms) and 50Hertz (2 farms: Wikinger, Arkona). Amprion and TransnetBW have no offshore capacity.
- SMARD `tso/` subdirectory separates per-TSO files from national files to avoid confusion.
- Each weather Parquet has columns like `temperature_2m_city_berlin`, `wind_speed_100m_won_hueselitz` — variable name + location suffix.

---

### 2.13 Milestone & Tests

**Tests** (`tests/`):

- **`test_smard_api.py`** — Mock the SMARD API (use `responses` or `pytest-httpserver`). Test:
  - `get_timestamps()` returns sorted list of millisecond timestamps
  - `get_data()` parses the JSON response into a DataFrame with correct columns
  - `get_all_data()` concatenates chunks and deduplicates
  - `DataNotAvailableError` raised on 404

- **`test_smard_source.py`** — Mock API responses, test SmardSource:
  - `download()` creates a Parquet file with expected columns for a national region
  - `download()` with TSO region produces correctly suffixed columns
  - `update()` with existing data only fetches the redundancy window
  - Resume support: columns already in the Parquet are skipped
  - `KNOWN_MISSING` combinations don't cause errors

- **`test_weather.py`** — Mock Open-Meteo API, test OpenMeteoSource:
  - `load_locations()` returns correct locations for each (type, TSO) pair
  - Physical validation clips out-of-range values
  - Solar elevation columns are added to archive data
  - Three files created with correct names

- **`test_commodities.py`** — Test reconstruction logic with synthetic data:
  - `reconstruct_ttf()`: given a Yahoo series with a gap and FRED data, the gap is filled. Bias correction adjusts the FRED-based fill to match Yahoo in the overlap. Output has no NaN in the gap period.
  - `merge_carbon()`: ICAP + CO2.L merged correctly. CO2.L values are bias-corrected. Forward-fill only within valid range.
  - Price range validation flags outliers.

- **`test_config_smard.py`** — Sanity checks on config:
  - All `KNOWN_MISSING` filter IDs exist in `TSO_FILTER_KEYS`
  - All `TSO_SUFFIXES` TSOs match `TSO_REGIONS`
  - `NATIONAL_REGIONS` values and `TSO_REGIONS` keys have no overlap

- **`test_cli.py`** — Smoke test CLI commands exist (use `typer.testing.CliRunner`):
  - `download --help` exits 0
  - `update --help` exits 0

- **`test_data_coverage.py`** — Run after a full download to validate data coverage and gaps. Not a unit test (requires real data on disk); mark with `@pytest.mark.slow` or put in a separate `tests/integration/` directory. Checks:
  - Each commodity Parquet exists and has data starting no later than its expected start date (e.g., Brent from 2015, TTF from 2015 after reconstruction, carbon from 2014)
  - No commodity has an unexpected multi-month gap (max gap < 30 calendar days, allowing for weekends/holidays)
  - Each SMARD national Parquet has the expected number of columns (spot-check)
  - Each SMARD per-TSO Parquet has columns matching the non-KNOWN_MISSING filter keys for that TSO
  - Weather Parquets have values within physical limits (sample check)
  - All Parquet DatetimeIndex values are timezone-aware UTC

**Milestone checklist:**

- [ ] `energy-forecasting download all` runs end-to-end (may take hours — verify with a subset first)
- [ ] `energy-forecasting update all` incrementally updates all sources
- [ ] `make data` triggers full download
- [ ] `make update` triggers incremental update
- [ ] SMARD national Parquets exist with expected columns (spot-check column count and date range)
- [ ] SMARD per-TSO Parquets have correctly suffixed column names
- [ ] Weather Parquets have physical-limit-validated values and solar elevation columns
- [ ] TTF reconstruction fills the Dec 2014-Oct 2017 gap with no NaN
- [ ] Carbon merge produces unified series through present day
- [ ] All Parquet files have continuous DatetimeIndex (hourly for SMARD/weather, daily for commodities)
- [ ] All Parquet files use zstd compression and reduced dtypes
- [ ] `make test` passes all stage 2 tests
- [ ] `make lint` passes

**Stage-gate verification** (from master plan): Raw data matches EP/EMA downloads. Specifically:
- SMARD DE_LU column values should match EP's `data/interim/combined_de_lu_hourly.parquet` within floating-point tolerance (we may have slightly different date ranges due to timing).
- Per-TSO data should match EMA's `database/DE/smard_v2/history_hourly.parquet` columns.
- Commodity prices should match EP's `data/interim/commodity_prices_daily.parquet`.

---

### Implementation Notes

**Estimated complexity by component:**
- `config/smard.py` + `config/commodities.py` — straightforward config porting
- `data/smard.py` — straightforward API client (~100 lines)
- `data/sources.py` DataSource base + SmardSource — moderate (~200 lines). The per-column merge and resume logic needs care.
- `data/weather.py` OpenMeteoSource — **most complex** (~300 lines). Three-endpoint merge, multi-location handling, solar elevation computation. Port carefully from EMA.
- `data/commodities.py` — moderate (~250 lines). TTF reconstruction and carbon merge are non-trivial but well-understood from EP.
- `cli.py` — straightforward Typer wiring

**Potential blockers:**
- SMARD API rate limits (undocumented). EP uses 8-10 parallel workers without issues. If we hit limits, reduce `max_workers`.
- Open-Meteo API load shedding. EMA handles this with `requests-cache` + `retry-requests`. The retry stack should handle transient failures, but a full download of all locations across 3 endpoints for 4 asset types will take significant time.
- FRED API key required for TTF reconstruction. Without it, the historical TTF gap cannot be filled. The key is free but requires registration at https://fred.stlouisfed.org/docs/api/api_key.html.
- ICAP API access — EP's ICAP download functions may require specific authentication or URL patterns that need to be verified.

**pyproject.toml entry point** — add to `[project.scripts]`:
```toml
[project.scripts]
energy-forecasting = "energy_forecasting.cli:app"
```

---
