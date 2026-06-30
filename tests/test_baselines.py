"""Tests for modeling/baselines.py."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.baselines import (
    naive_lag,
    naive_seasonal_7d,
    naive_weekly,
)


@pytest.fixture
def hourly_series():
    """30 days of hourly data with a pattern."""
    idx = pd.date_range("2024-01-01", periods=30 * 24, freq="h")
    # Simple pattern: hour of day + noise
    values = np.tile(np.arange(24, dtype=float), 30)
    return pd.Series(values, index=idx)


class TestNaiveLag:
    def test_default_24h(self, hourly_series):
        result = naive_lag(hourly_series)
        # First 24 values should be NaN
        assert result.iloc[:24].isna().all()
        # After that, value should equal value from 24 hours ago
        assert result.iloc[24] == hourly_series.iloc[0]

    def test_custom_lag(self, hourly_series):
        result = naive_lag(hourly_series, lag=48)
        assert result.iloc[:48].isna().all()
        assert result.iloc[48] == hourly_series.iloc[0]

    def test_preserves_index(self, hourly_series):
        result = naive_lag(hourly_series)
        pd.testing.assert_index_equal(result.index, hourly_series.index)


class TestNaiveWeekly:
    def test_168h_lag(self, hourly_series):
        result = naive_weekly(hourly_series)
        assert result.iloc[:168].isna().all()
        assert result.iloc[168] == hourly_series.iloc[0]


class TestNaiveSeasonal7d:
    def test_averages_past_weeks(self, hourly_series):
        result = naive_seasonal_7d(hourly_series, n_weeks=2)
        # First week should be NaN (no prior data)
        assert result.iloc[:168].isna().all()
        # After 2 weeks, should have values
        assert not result.iloc[2 * 168:].isna().all()

    def test_shape_preserved(self, hourly_series):
        result = naive_seasonal_7d(hourly_series)
        assert len(result) == len(hourly_series)
