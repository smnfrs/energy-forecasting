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
        existing data -- the merge logic handles deduplication.
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
            f"Saved {self.output_path.name}: {len(df)} rows, {df.index.min()} to {df.index.max()}"
        )

    def update(self) -> None:
        """Incremental update. Fetches new data and merges with existing."""
        if not self.output_path.exists():
            logger.info("No existing data, running full download")
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


# ── SmardSource ─────────────────────────────────────────────────────


class SmardSource:
    """SMARD data for a single region (national or TSO).

    Unlike DataSource subclasses, SmardSource manages per-column
    merging (one API call per filter key), so it implements
    download/update directly rather than fetch_all/fetch_update.

    National regions (DE-LU, DE-AT-LU): generation, load, prices,
    forecasts, cross-border flows -- all filter keys for that region.

    TSO regions (50Hertz, Amprion, etc.): generation and load only,
    using the per-TSO filter keys from config/smard.py.

    Each region produces one Parquet file with columns named by
    clean_column_name() for national, or by TSO_FILTER_KEYS values
    with TSO suffix for per-TSO.
    """

    def __init__(self, region: str, resolution: str = "hour"):
        from energy_forecasting.config.smard import TSO_REGIONS

        self.region = region
        self.resolution = resolution
        self._is_tso = region in TSO_REGIONS

    @property
    def output_path(self) -> Path:
        from energy_forecasting.config import SMARD_DIR

        if self._is_tso:
            return SMARD_DIR / "tso" / f"{self.region}.parquet"
        return SMARD_DIR / f"{self.region}.parquet"

    @property
    def filter_keys(self) -> dict[int, str]:
        """Filter keys valid for this region."""
        from energy_forecasting.config.columns import (
            CROSS_BORDER_DE_AT_LU,
            CROSS_BORDER_DE_LU,
            EXCLUDED_KEYS,
            SMARD_FILTER_KEYS,
        )
        from energy_forecasting.config.smard import KNOWN_MISSING, TSO_FILTER_KEYS

        if self._is_tso:
            return {
                k: v for k, v in TSO_FILTER_KEYS.items() if (k, self.region) not in KNOWN_MISSING
            }
        # National region: combine SMARD national keys with cross-border flows
        flow_dict = CROSS_BORDER_DE_LU if self.region == "DE-LU" else CROSS_BORDER_DE_AT_LU
        combined = {k: v for k, v in SMARD_FILTER_KEYS.items() if k not in EXCLUDED_KEYS}
        combined.update(flow_dict)
        return combined

    def _column_name(self, filter_id: int, base_name: str) -> str:
        """Column name for a given filter in this region."""
        from energy_forecasting.config.columns import SMARD_COLUMN_NAMES, clean_column_name
        from energy_forecasting.config.smard import TSO_SUFFIXES

        if self._is_tso:
            return f"{base_name}{TSO_SUFFIXES[self.region]}"
        # SMARD_COLUMN_NAMES covers SMARD_FILTER_KEYS; cross-border flow keys
        # have their own descriptions that need cleaning.
        return SMARD_COLUMN_NAMES.get(filter_id, clean_column_name(base_name))

    def download(self) -> None:
        """Download all filter keys for this region.

        Uses ThreadPoolExecutor for parallel API calls, then
        merges columns sequentially into a single Parquet file.
        Crash-resilient: existing columns in the output file are
        skipped on resume (EMA pattern).
        """
        import requests as req_lib
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from energy_forecasting.data.smard import DataNotAvailableError, get_all_data

        logger.info(f"Downloading SMARD {self.region} ({len(self.filter_keys)} filter keys)")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing columns (resume support)
        existing_cols: set[str] = set()
        if self.output_path.exists():
            existing_cols = set(load_parquet(self.output_path).columns)

        # Shared session for connection pooling across all workers
        session = req_lib.Session()

        def fetch_one(filter_id: int, name: str) -> tuple[str, pd.DataFrame]:
            col = self._column_name(filter_id, name)
            if col in existing_cols:
                logger.debug(f"Skipping {col} (already exists)")
                return col, pd.DataFrame()
            try:
                df = get_all_data(filter_id, self.region, self.resolution, session=session)
                df = df.rename(columns={"value": col})
                return col, df
            except DataNotAvailableError:
                logger.warning(f"No data: {filter_id}/{self.region}")
                return col, pd.DataFrame()

        results: dict[str, pd.Series] = {}
        from energy_forecasting.config.smard import MAX_SMARD_WORKERS

        with ThreadPoolExecutor(max_workers=MAX_SMARD_WORKERS) as pool:
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
            f"Saved {self.output_path.name}: {len(combined)} rows, {len(combined.columns)} columns"
        )

    def update(self) -> None:
        """Incremental update with redundancy window.

        SMARD data comes in weekly chunks identified by millisecond timestamps.
        1. Fetch timestamp index from API (one call per filter key)
        2. Convert local cutoff (last_ts - redundancy) to milliseconds
        3. bisect_left to find starting chunk index
        4. Download chunks from that index forward (parallel)
        5. Merge into existing Parquet (per-column, keep-new on overlap)
        """
        import bisect
        import requests as req_lib
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from energy_forecasting.config.smard import (
            BOOTSTRAP_DAYS,
            DEFAULT_REDUNDANCY_DAYS,
            MAX_SMARD_WORKERS,
        )
        from energy_forecasting.data.smard import (
            DataNotAvailableError,
            get_all_data,
            get_timestamps,
        )

        if not self.output_path.exists():
            self.download()
            return

        existing = load_parquet(self.output_path)
        existing_cols = set(existing.columns)

        # Shared session for connection pooling across all workers
        session = req_lib.Session()

        # Identify new columns to bootstrap vs existing columns to update
        all_filter_items = list(self.filter_keys.items())
        missing_items = [
            (fid, name)
            for fid, name in all_filter_items
            if self._column_name(fid, name) not in existing_cols
        ]
        update_items = [
            (fid, name)
            for fid, name in all_filter_items
            if self._column_name(fid, name) in existing_cols
        ]

        # Bootstrap missing columns (last BOOTSTRAP_DAYS days only)
        if missing_items:
            logger.info(
                f"Bootstrapping {len(missing_items)} new columns from last {BOOTSTRAP_DAYS} days"
            )
            bootstrap_cutoff_ms = int(
                (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=BOOTSTRAP_DAYS)).timestamp() * 1000
            )
            for fid, name in missing_items:
                try:
                    all_ts = get_timestamps(fid, self.region, self.resolution, session=session)
                    start_idx = bisect.bisect_left(all_ts, bootstrap_cutoff_ms)
                    df = get_all_data(
                        fid,
                        self.region,
                        self.resolution,
                        timestamp_list=all_ts[start_idx:],
                        session=session,
                    )
                    if not df.empty:
                        col = self._column_name(fid, name)
                        existing = _merge_column(existing, col, df["value"].rename(col))
                except DataNotAvailableError:
                    pass

        # Incremental update for existing columns
        last_ts = existing.index.max()
        cutoff_ms = int((last_ts - pd.Timedelta(days=DEFAULT_REDUNDANCY_DAYS)).timestamp() * 1000)

        def fetch_update_one(fid: int, name: str) -> tuple[str, pd.Series]:
            col = self._column_name(fid, name)
            try:
                all_ts = get_timestamps(fid, self.region, self.resolution, session=session)
                start_idx = bisect.bisect_left(all_ts, cutoff_ms)
                df = get_all_data(
                    fid,
                    self.region,
                    self.resolution,
                    timestamp_list=all_ts[start_idx:],
                    session=session,
                )
                if not df.empty:
                    return col, df["value"].rename(col)
            except DataNotAvailableError:
                pass
            return col, pd.Series(dtype=float)

        with ThreadPoolExecutor(max_workers=MAX_SMARD_WORKERS) as pool:
            futures = {
                pool.submit(fetch_update_one, fid, name): (fid, name) for fid, name in update_items
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



# ── EnergyChartsSource ──────────────────────────────────────────────


class EnergyChartsSource(DataSource):
    """Day-ahead prices from energy-charts.info.

    Used as a fallback when SMARD has gaps. Not critical for
    normal operation.
    """

    def __init__(self, series_name: str = "da_price_de_lu"):
        from energy_forecasting.config.commodities import ENERGY_CHARTS_SERIES

        self.config = ENERGY_CHARTS_SERIES[series_name]
        self._name = series_name

    @property
    def output_path(self) -> Path:
        from energy_forecasting.config import ENERGY_CHARTS_DIR

        return ENERGY_CHARTS_DIR / f"{self._name}.parquet"

    def _fetch_price(self, bzn: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch day-ahead prices from Energy Charts API."""
        import requests

        from energy_forecasting.config.commodities import ENERGY_CHARTS_BASE_URL

        params = {
            "bzn": bzn,
            "start": start.strftime("%Y-%m-%dT%H:%M+00:00"),
            "end": end.strftime("%Y-%m-%dT%H:%M+00:00"),
        }
        resp = requests.get(f"{ENERGY_CHARTS_BASE_URL}/price", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
                "value": data["price"],
            }
        )
        df = df.set_index("time")
        df = df.dropna(subset=["value"])
        df = df.rename(columns={"value": self.config["column"]})
        return df

    def fetch_all(self) -> pd.DataFrame:
        """Fetch full history from Energy Charts API."""
        bzn = self.config["bzn"]
        start = pd.Timestamp("2015-01-01", tz="UTC")
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=2)
        return self._fetch_price(bzn, start, end)

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        """Fetch from last_timestamp to tomorrow+2."""
        bzn = self.config["bzn"]
        start = last_timestamp - pd.Timedelta(days=7)
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=2)
        return self._fetch_price(bzn, start, end)


# ── Helpers ─────────────────────────────────────────────────────────


def _merge_column(df: pd.DataFrame, col_name: str, series: pd.Series) -> pd.DataFrame:
    """Merge a single column into an existing DataFrame.

    New timestamps extend the index. Overlapping timestamps are
    overwritten (keep new).
    """
    if col_name in df.columns:
        combined_index = df.index.union(series.index)
        df = df.reindex(combined_index)
        df.loc[series.index, col_name] = series.values
    else:
        combined_index = df.index.union(series.index)
        df = df.reindex(combined_index)
        df[col_name] = series.reindex(combined_index)
    return df
