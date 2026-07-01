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
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from energy_forecasting.config import COMMODITIES_DIR
from energy_forecasting.config.commodities import (
    COLUMN_NAMES,
    DATA_START,
    FRED_SERIES,
    ICAP_SYSTEMS,
    ICAP_URL,
    PRICE_RANGES,
    TICKERS,
    TTF_RECONSTRUCTION_START,
    TTF_YAHOO_START,
    UNIT_CONVERSION_MMBTU_TO_MWH,
)
from energy_forecasting.data.io import load_parquet
from energy_forecasting.data.sources import DataSource

# ── Individual source downloads ─────────────────────────────────────


def _download_icap() -> pd.DataFrame:
    """Fetch ICAP carbon data for both ETS phases.

    Downloads Phase 3 (2014-2018) and Phase 4 (2019+), concatenates,
    and returns a DataFrame with columns: carbon_primary, carbon_secondary,
    eur_usd_rate.
    """
    frames = []
    for phase_name, system_id in ICAP_SYSTEMS.items():
        start_date = DATA_START["carbon_icap"]
        params = {
            "systemIds": system_id,
            "startDate": int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000),
            "endDate": int(pd.Timestamp.now(tz="UTC").timestamp() * 1000),
        }
        resp = requests.get(ICAP_URL, params=params, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text), skiprows=1)

        # Clean up: remove unnamed columns and trailing spaces
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()

        # Parse date column
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # Select and rename columns
        columns_to_keep = {}
        if "Primary Market" in df.columns:
            columns_to_keep["Primary Market"] = "carbon_primary"
        if "Secondary Market" in df.columns:
            columns_to_keep["Secondary Market"] = "carbon_secondary"
        if "Exchange rate EUR/USD" in df.columns:
            columns_to_keep["Exchange rate EUR/USD"] = "eur_usd_rate"

        df = df[list(columns_to_keep.keys())].copy()
        df.columns = list(columns_to_keep.values())

        # Convert to numeric (handles empty strings as NaN)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        frames.append(df)

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined.index.name = "date"
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

    eur_usd_daily = icap["eur_usd_rate"].reindex(fred_us.index, method="ffill")
    us_gas_eur = (fred_us / eur_usd_daily) * UNIT_CONVERSION_MMBTU_TO_MWH

    # Step 2: Bias correction (EU adjusted to match Yahoo in overlap)
    overlap_start = pd.Timestamp(TTF_YAHOO_START, tz="UTC")
    eu_overlap = eu_gas_daily.loc[overlap_start:]
    yahoo_overlap = ttf_yahoo.reindex(eu_overlap.index)
    valid = eu_overlap.notna() & yahoo_overlap.notna()
    if valid.any():
        bias = (yahoo_overlap[valid] - eu_overlap[valid]).mean()
        corr = yahoo_overlap[valid].corr(eu_overlap[valid])
        logger.info(f"TTF bias correction: {bias:.2f} EUR/MWh, r={corr:.3f}")
    else:
        bias = 0.0
    eu_adjusted = eu_gas_daily + bias

    # Step 3: US daily variation (centered -- removes level, keeps shape)
    us_monthly_mean = us_gas_eur.resample("MS").mean()
    us_centered = us_gas_eur - us_monthly_mean.reindex(us_gas_eur.index, method="ffill")

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

    # Validate (flag only, do not clip)
    lo, hi = PRICE_RANGES["ttf"]
    outliers = (full < lo) | (full > hi)
    if outliers.any():
        logger.warning(f"TTF: {outliers.sum()} values outside [{lo}, {hi}]")

    return full


def merge_carbon(raw_dir: Path) -> pd.DataFrame:
    """Merge ICAP historical carbon with CO2.L realtime.

    ICAP is official but has ~60-day publication lag.
    CO2.L (SparkChange EUA ETC) is realtime but only from Oct 2021.
    Bias-correct CO2.L to match ICAP in the overlap period, then
    use CO2.L to extend forward where ICAP is missing.

    Returns: DataFrame with column carbon_eur_per_ton and daily DatetimeIndex.
    The realtime column is used internally for gap-filling but dropped from
    output (matching EP's approach — it adds no signal beyond the unified column).
    Ported from EP's src/data/commodities.py (~line 682-820).
    """
    icap = load_parquet(raw_dir / "icap.parquet")
    co2l = load_parquet(raw_dir / "carbon_realtime.parquet")

    carbon_col = COLUMN_NAMES["carbon"]
    carbon_rt_col = COLUMN_NAMES["carbon_realtime"]

    # ICAP: use primary market price
    carbon = icap["carbon_primary"].rename(carbon_col)
    co2l_price = co2l["price"].rename(carbon_rt_col)

    # Merge on date
    combined = pd.concat([carbon, co2l_price], axis=1)

    # Bias correction in overlap period
    overlap = combined.dropna()
    if len(overlap) > 0:
        bias = (overlap[carbon_col] - overlap[carbon_rt_col]).mean()
        corr = overlap[carbon_col].corr(overlap[carbon_rt_col])
        logger.info(f"Carbon bias: {bias:.2f} EUR/ton, r={corr:.3f}")
    else:
        bias = 0.0

    # Fill ICAP gaps with bias-corrected CO2.L
    missing_icap = combined[carbon_col].isna()
    combined.loc[missing_icap, carbon_col] = combined.loc[missing_icap, carbon_rt_col] + bias

    # Forward-fill within valid data range only (bounded ffill)
    for col in combined.columns:
        first_valid = combined[col].first_valid_index()
        last_valid = combined[col].last_valid_index()
        if first_valid is not None:
            mask = (combined.index >= first_valid) & (combined.index <= last_valid)
            combined.loc[mask, col] = combined.loc[mask, col].ffill()

    # Drop realtime column — it was only used to extend the unified carbon series.
    # Keeping it would add a 60% NaN column with no additional signal beyond
    # what carbon_eur_per_ton already captures.
    combined = combined.drop(columns=[carbon_rt_col])

    # Validate (flag only, do not clip)
    lo, hi = PRICE_RANGES["carbon"]
    outliers = (combined[carbon_col] < lo) | (combined[carbon_col] > hi)
    if outliers.any():
        logger.warning(f"Carbon: {outliers.sum()} values outside [{lo}, {hi}]")

    return combined


# ── DataSource subclasses ───────────────────────────────────────────


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

    UPDATE_OVERLAP_DAYS = 7

    def __init__(self, ticker_key: str):
        self.ticker_key = ticker_key

    @property
    def output_path(self) -> Path:
        return COMMODITIES_DIR / f"{self.ticker_key}.parquet"

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize yfinance output to price/volume with UTC index."""
        # yfinance >=0.2.31 returns MultiIndex columns (metric, ticker).
        # Flatten to single level before selecting.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Close", "Volume"]].rename(columns={"Close": "price", "Volume": "volume"})
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df

    def fetch_all(self) -> pd.DataFrame:
        import yfinance as yf

        ticker = TICKERS[self.ticker_key]
        start = DATA_START.get(f"{self.ticker_key}_yahoo", DATA_START.get(self.ticker_key))
        df = yf.download(ticker, start=str(start), progress=False)
        if df.empty:
            return df
        return self._normalize(df)

    def fetch_update(self, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        import yfinance as yf

        ticker = TICKERS[self.ticker_key]
        start = (last_timestamp - pd.Timedelta(days=self.UPDATE_OVERLAP_DAYS)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, progress=False)
        if df.empty:
            return df
        return self._normalize(df)


class FredSource(DataSource):
    """FRED gas price series. Requires FRED_API_KEY env var."""

    def __init__(self, series_key: str):
        self.series_key = series_key

    @property
    def output_path(self) -> Path:
        return COMMODITIES_DIR / f"{self.series_key}.parquet"

    def fetch_all(self) -> pd.DataFrame:
        from fredapi import Fred

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


# ── All commodity sources (for CLI iteration) ───────────────────────


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
