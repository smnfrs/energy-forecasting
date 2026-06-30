"""Tests for data/commodities.py — reconstruction logic with synthetic data."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.config.commodities import COLUMN_NAMES, PRICE_RANGES
from energy_forecasting.data.commodities import YahooSource, merge_carbon, reconstruct_ttf
from energy_forecasting.data.io import save_parquet

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def raw_dir(tmp_path):
    """Create a raw_dir with synthetic commodity data for reconstruction tests."""
    # TTF Yahoo data (starts 2017-10-23, daily)
    ttf_idx = pd.date_range("2017-10-23", periods=200, freq="B", tz="UTC")
    ttf_df = pd.DataFrame(
        {"price": np.random.default_rng(42).uniform(15, 30, len(ttf_idx))}, index=ttf_idx
    )
    ttf_df.index.name = "date"
    save_parquet(ttf_df, tmp_path / "ttf.parquet")

    # FRED EU gas (monthly, starts 2014)
    fred_eu_idx = pd.date_range("2014-01-01", "2018-12-01", freq="MS", tz="UTC")
    fred_eu_df = pd.DataFrame(
        {"price": np.random.default_rng(43).uniform(3, 8, len(fred_eu_idx))},
        index=fred_eu_idx,
    )
    fred_eu_df.index.name = "date"
    save_parquet(fred_eu_df, tmp_path / "fred_eu_gas.parquet")

    # FRED US gas (daily, starts 2014)
    fred_us_idx = pd.date_range("2014-01-01", "2018-12-31", freq="B", tz="UTC")
    fred_us_df = pd.DataFrame(
        {"price": np.random.default_rng(44).uniform(2, 5, len(fred_us_idx))},
        index=fred_us_idx,
    )
    fred_us_df.index.name = "date"
    save_parquet(fred_us_df, tmp_path / "fred_us_gas.parquet")

    # ICAP carbon (daily, with eur_usd_rate)
    icap_idx = pd.date_range("2014-11-03", "2018-12-31", freq="B", tz="UTC")
    icap_df = pd.DataFrame(
        {
            "carbon_primary": np.random.default_rng(45).uniform(5, 25, len(icap_idx)),
            "carbon_secondary": np.random.default_rng(46).uniform(5, 25, len(icap_idx)),
            "eur_usd_rate": np.random.default_rng(47).uniform(1.05, 1.20, len(icap_idx)),
        },
        index=icap_idx,
    )
    icap_df.index.name = "date"
    save_parquet(icap_df, tmp_path / "icap.parquet")

    # CO2.L realtime (daily, starts 2021-10-18)
    co2l_idx = pd.date_range("2021-10-18", periods=100, freq="B", tz="UTC")
    co2l_df = pd.DataFrame(
        {"price": np.random.default_rng(48).uniform(50, 90, len(co2l_idx))},
        index=co2l_idx,
    )
    co2l_df.index.name = "date"
    save_parquet(co2l_df, tmp_path / "carbon_realtime.parquet")

    return tmp_path


# ── TTF reconstruction tests ───────────────────────────────────────


def test_reconstruct_ttf_fills_gap(raw_dir):
    result = reconstruct_ttf(raw_dir)
    assert isinstance(result, pd.Series)
    assert result.name == COLUMN_NAMES["ttf"]

    # Gap period should have data (Dec 2014 to Oct 2017)
    gap_data = result.loc["2015-01-01":"2017-10-22"]
    assert len(gap_data) > 0
    assert gap_data.notna().any(), "Gap period should have reconstructed data"


def test_reconstruct_ttf_preserves_yahoo(raw_dir):
    result = reconstruct_ttf(raw_dir)
    ttf_yahoo = pd.read_parquet(raw_dir / "ttf.parquet")["price"]
    # Yahoo values should be preserved (kept as 'last' in dedup)
    overlap = result.reindex(ttf_yahoo.index)
    pd.testing.assert_series_equal(
        overlap.dropna(),
        ttf_yahoo.dropna().rename(COLUMN_NAMES["ttf"]),
        check_names=False,
    )


def test_reconstruct_ttf_sorted_index(raw_dir):
    result = reconstruct_ttf(raw_dir)
    assert result.index.is_monotonic_increasing


# ── Carbon merge tests ──────────────────────────────────────────────


def test_merge_carbon_has_expected_columns(raw_dir):
    result = merge_carbon(raw_dir)
    assert COLUMN_NAMES["carbon"] in result.columns
    # carbon_realtime is used internally but dropped from output
    assert COLUMN_NAMES["carbon_realtime"] not in result.columns


def test_merge_carbon_extends_icap(raw_dir):
    """CO2.L data should extend the carbon series beyond ICAP's end."""
    result = merge_carbon(raw_dir)
    carbon_col = COLUMN_NAMES["carbon"]
    # Should have data in the CO2.L period (after ICAP ends)
    co2l_period = result.loc["2021-10-18":]
    assert co2l_period[carbon_col].notna().any()


def test_merge_carbon_bounded_ffill(raw_dir):
    result = merge_carbon(raw_dir)
    carbon_col = COLUMN_NAMES["carbon"]
    # Data before first valid should be NaN (not forward-filled)
    first_valid = result[carbon_col].first_valid_index()
    if first_valid is not None and first_valid > result.index[0]:
        before_first = result.loc[:first_valid].iloc[:-1]
        assert before_first[carbon_col].isna().all()


# ── Price range validation ──────────────────────────────────────────


def test_price_ranges_defined():
    assert "carbon" in PRICE_RANGES
    assert "ttf" in PRICE_RANGES
    assert "brent" in PRICE_RANGES
    for key, (lo, hi) in PRICE_RANGES.items():
        assert lo < hi, f"Invalid range for {key}: [{lo}, {hi}]"


# ── YahooSource._normalize tests ──────────────────────────────────


def test_normalize_flattens_multiindex_columns():
    """yfinance >=0.2.31 returns MultiIndex (metric, ticker). _normalize must flatten."""
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            ("Close", "BZ=F"): [70.0, 71.0, 72.0, 73.0, 74.0],
            ("High", "BZ=F"): [71.0, 72.0, 73.0, 74.0, 75.0],
            ("Low", "BZ=F"): [69.0, 70.0, 71.0, 72.0, 73.0],
            ("Open", "BZ=F"): [70.5, 71.5, 72.5, 73.5, 74.5],
            ("Volume", "BZ=F"): [1000, 2000, 3000, 4000, 5000],
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    result = YahooSource._normalize(df)
    assert result.columns.tolist() == ["price", "volume"]
    assert not isinstance(result.columns, pd.MultiIndex)


def test_normalize_handles_flat_columns():
    """Older yfinance versions return flat columns. _normalize must still work."""
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Close": [70.0, 71.0, 72.0, 73.0, 74.0],
            "High": [71.0, 72.0, 73.0, 74.0, 75.0],
            "Volume": [1000, 2000, 3000, 4000, 5000],
        },
        index=idx,
    )

    result = YahooSource._normalize(df)
    assert result.columns.tolist() == ["price", "volume"]


def test_normalize_output_index_is_utc():
    """_normalize must produce a UTC-aware DatetimeIndex named 'date'."""
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [10, 20, 30]}, index=idx)

    result = YahooSource._normalize(df)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"
    assert result.index.name == "date"


def test_yahoo_sources_return_distinct_data():
    """Each Yahoo ticker must produce different data.

    Catches yfinance thread-safety regressions: when called concurrently
    all tickers returned identical data. Even when called sequentially,
    this test verifies each source fetches its own ticker.
    """
    sources = [YahooSource("ttf"), YahooSource("brent"), YahooSource("carbon_realtime")]
    results = {}
    for source in sources:
        df = source.fetch_all()
        if df.empty:
            pytest.skip(f"No data returned for {source.ticker_key} (network issue?)")
        results[source.ticker_key] = df

    # Each pair must differ in either index range or values
    keys = list(results.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            df1, df2 = results[k1], results[k2]
            same_shape = df1.shape == df2.shape
            if same_shape:
                same_values = (df1["price"].values == df2["price"].values).all()
                assert not same_values, (
                    f"{k1} and {k2} returned identical price data — "
                    f"yfinance likely corrupted by concurrent access"
                )
