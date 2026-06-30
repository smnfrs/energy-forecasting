"""Tests for features/market.py — market feature functions."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.features.market import (
    compute_day_index,
    compute_ewma,
    compute_fourier_features,
    compute_german_holidays,
    compute_hourly_lag,
    compute_interaction,
    compute_rolling_stat,
    compute_temporal_features,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def hourly_index():
    """7 days of hourly data, UTC."""
    return pd.date_range("2024-01-08", periods=168, freq="h", tz="UTC")


@pytest.fixture
def hourly_series(hourly_index):
    """Simple increasing series for testing."""
    return pd.Series(np.arange(168, dtype=float), index=hourly_index)


@pytest.fixture
def price_like(hourly_index):
    """Realistic-ish price series with daily seasonality."""
    rng = np.random.default_rng(42)
    return pd.Series(
        50 + 20 * np.sin(np.arange(168) * 2 * np.pi / 24) + rng.normal(0, 5, 168),
        index=hourly_index,
    )


# ── compute_hourly_lag ────────────────────────────────────────────


def test_hourly_lag_shift():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    s = pd.Series(range(48), index=idx, dtype=float)
    result = compute_hourly_lag(s, 24)
    assert result.iloc[23] != result.iloc[23]  # NaN
    assert result.iloc[24] == 0.0  # first valid = value from 24h ago


def test_hourly_lag_168():
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    s = pd.Series(range(200), index=idx, dtype=float)
    result = compute_hourly_lag(s, 168)
    assert pd.isna(result.iloc[167])
    assert result.iloc[168] == 0.0


# ── compute_rolling_stat ──────────────────────────────────────────


def test_rolling_stat_avg(hourly_series):
    """Week-rolling avg over D-7 to D-1 should give a scalar per day."""
    result = compute_rolling_stat(hourly_series, start_day=-7, end_day=-1, stat="avg")
    # Last day (Jan 14) should have a valid rolling avg
    jan14 = result.loc["2024-01-14"]
    assert jan14.notna().all()
    # All hours of a day should get the same value (daily broadcast)
    assert jan14.nunique() == 1


def test_rolling_stat_std(price_like):
    result = compute_rolling_stat(price_like, start_day=-7, end_day=-1, stat="std")
    # Std should be > 0 for non-constant data
    valid = result.dropna()
    assert (valid > 0).all()


def test_rolling_stat_single_day(hourly_series):
    """_d2 → start_day=-2, end_day=-2, avg of day before yesterday."""
    result = compute_rolling_stat(hourly_series, start_day=-2, end_day=-2, stat="avg")
    # Day index 2 (Jan 10) should have avg of Jan 8 (index 0-23)
    jan10 = result.loc["2024-01-10"]
    assert jan10.notna().all()
    expected = np.mean(range(24))  # avg of 0..23 = 11.5
    assert abs(jan10.iloc[0] - expected) < 0.01


def test_rolling_stat_with_end_hour(hourly_series):
    """end_hour=10 should only use hours 0-9 of the final day."""
    result = compute_rolling_stat(hourly_series, start_day=-1, end_day=-1, stat="avg", end_hour=10)
    # For Jan 9 (day 2), D-1=Jan 8, hours 0-9 → values 0..9, avg=4.5
    jan09 = result.loc["2024-01-09"]
    assert jan09.notna().all()
    expected = np.mean(range(10))  # 4.5
    assert abs(jan09.iloc[0] - expected) < 0.01


def test_rolling_stat_with_hour_filter(price_like):
    """_h8_h20 should only use hours 8-19."""
    result = compute_rolling_stat(
        price_like, start_day=-7, end_day=-1, stat="avg", hour_start=8, hour_end=20
    )
    valid = result.dropna()
    assert len(valid) > 0


def test_rolling_stat_range(hourly_series):
    result = compute_rolling_stat(hourly_series, start_day=-1, end_day=-1, stat="range")
    # Day 2 (Jan 9), D-1=Jan 8, values 0..23, range=23
    jan09 = result.loc["2024-01-09"]
    assert jan09.notna().all()
    assert abs(jan09.iloc[0] - 23.0) < 0.01


def test_daily_broadcast(hourly_series):
    """start_day=0, end_day=0 → current-day broadcast aggregate."""
    result = compute_rolling_stat(hourly_series, start_day=0, end_day=0, stat="avg")
    # All hours of each day should get that day's average
    jan08 = result.loc["2024-01-08"]
    expected = np.mean(range(24))  # 11.5
    assert abs(jan08.iloc[0] - expected) < 0.01


# ── compute_ewma ──────────────────────────────────────────────────


def test_ewma_no_cutoff(price_like):
    result = compute_ewma(price_like, span=6)
    assert len(result) == len(price_like)
    assert result.notna().sum() > 0


def test_ewma_with_cutoff(price_like):
    result = compute_ewma(price_like, span=6, cutoff_day=-1)
    # All hours of a day should get the same EWMA value (broadcast)
    day = result.loc["2024-01-12"]
    valid = day.dropna()
    if len(valid) > 1:
        assert valid.nunique() == 1


def test_ewma_with_cutoff_hour(price_like):
    result = compute_ewma(price_like, span=6, cutoff_day=-1, cutoff_hour=10)
    day = result.loc["2024-01-12"]
    valid = day.dropna()
    if len(valid) > 1:
        assert valid.nunique() == 1


# ── compute_temporal_features ─────────────────────────────────────


def test_temporal_features_columns(hourly_index):
    result = compute_temporal_features(hourly_index)
    expected_cols = {
        "_derived_hour_sin",
        "_derived_hour_cos",
        "_derived_dow_sin",
        "_derived_dow_cos",
        "_derived_month_sin",
        "_derived_month_cos",
        "_derived_is_weekend",
        "_derived_is_holiday",
        "_derived_day_index",
        "_derived_year_index",
    }
    assert expected_cols.issubset(set(result.columns))


def test_temporal_cyclical_range(hourly_index):
    result = compute_temporal_features(hourly_index)
    for col in ["_derived_hour_sin", "_derived_hour_cos"]:
        assert result[col].min() >= -1.0
        assert result[col].max() <= 1.0


def test_weekend_detection(hourly_index):
    result = compute_temporal_features(hourly_index)
    # Jan 8 2024 is Monday → weekday, Jan 13 2024 is Saturday → weekend
    assert result.loc["2024-01-08", "_derived_is_weekend"].iloc[0] == 0.0
    assert result.loc["2024-01-13", "_derived_is_weekend"].iloc[0] == 1.0


def test_german_holidays():
    idx = pd.date_range("2024-12-24", "2024-12-26", freq="h", tz="UTC")
    result = compute_german_holidays(idx)
    # Dec 25 and 26 are national holidays → 1.0
    dec25 = result.loc["2024-12-25"]
    assert (dec25 == 1.0).all()


# ── compute_fourier_features ──────────────────────────────────────


def test_fourier_column_count(hourly_index):
    result = compute_fourier_features(hourly_index, period=24, order=3)
    assert result.shape[1] == 6  # 3 sin + 3 cos


def test_fourier_values_bounded(hourly_index):
    result = compute_fourier_features(hourly_index, period=24, order=3)
    for col in result.columns:
        assert result[col].min() >= -1.0
        assert result[col].max() <= 1.0


# ── compute_interaction ───────────────────────────────────────────


def test_interaction_product():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    a = pd.Series([1, 2, 3, 4, 5], index=idx, dtype=float)
    b = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
    result = compute_interaction(a, b)
    expected = pd.Series([10, 40, 90, 160, 250], index=idx, dtype=float)
    pd.testing.assert_series_equal(result, expected)


# ── compute_day_index ─────────────────────────────────────────────


def test_day_index_epoch():
    idx = pd.DatetimeIndex(["2015-01-05"], tz="UTC")
    result = compute_day_index(idx)
    assert result.iloc[0] == 0

    idx2 = pd.DatetimeIndex(["2015-01-06"], tz="UTC")
    result2 = compute_day_index(idx2)
    assert result2.iloc[0] == 1
