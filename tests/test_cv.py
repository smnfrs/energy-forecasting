"""Tests for modeling/cv.py."""

import pandas as pd
import pytest
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout


@pytest.fixture
def hourly_index_1y():
    """1 year of hourly data (365 days, 8760 hours)."""
    return pd.date_range("2023-01-01", periods=365 * 24, freq="h")


class TestTimeSeriesSplitter:
    def test_basic_expanding(self, hourly_index_1y):
        splitter = TimeSeriesSplitter(n_splits=3)
        folds = list(splitter.split(hourly_index_1y))
        assert len(folds) == 3

        # Each fold has train and test
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_expanding_train_grows(self, hourly_index_1y):
        splitter = TimeSeriesSplitter(n_splits=3)
        folds = list(splitter.split(hourly_index_1y))
        # In expanding mode, train should grow (or at least not shrink)
        train_sizes = [len(t) for t, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_day_boundaries(self, hourly_index_1y):
        splitter = TimeSeriesSplitter(n_splits=3)
        folds = list(splitter.split(hourly_index_1y))
        for train_idx, test_idx in folds:
            # Train end should be at hour 23
            train_end_hour = hourly_index_1y[train_idx[-1]].hour
            assert train_end_hour == 23, f"Train end hour: {train_end_hour}"
            # Test start should be at hour 0
            test_start_hour = hourly_index_1y[test_idx[0]].hour
            assert test_start_hour == 0, f"Test start hour: {test_start_hour}"

    def test_fixed_test_days(self, hourly_index_1y):
        splitter = TimeSeriesSplitter(n_splits=3, test_days=30)
        folds = list(splitter.split(hourly_index_1y))
        for _, test_idx in folds:
            test_hours = len(test_idx)
            assert test_hours == 30 * 24

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            TimeSeriesSplitter(n_splits=3, mode="invalid")

    def test_min_splits(self):
        with pytest.raises(ValueError, match="n_splits"):
            TimeSeriesSplitter(n_splits=1)

    def test_get_n_splits(self):
        splitter = TimeSeriesSplitter(n_splits=5)
        assert splitter.get_n_splits() == 5

    def test_sliding_mode(self, hourly_index_1y):
        splitter = TimeSeriesSplitter(n_splits=3, mode="sliding")
        folds = list(splitter.split(hourly_index_1y))
        assert len(folds) >= 1
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestCarveHoldout:
    def test_basic_split(self, hourly_index_1y):
        pool_idx, holdout_idx = carve_holdout(hourly_index_1y, holdout_days=90)
        # Holdout should be last 90 days
        assert len(holdout_idx) == 90 * 24
        # Pool + holdout = total
        assert len(pool_idx) + len(holdout_idx) == len(hourly_index_1y)
        # No overlap
        assert len(set(pool_idx) & set(holdout_idx)) == 0

    def test_holdout_at_end(self, hourly_index_1y):
        pool_idx, holdout_idx = carve_holdout(hourly_index_1y, holdout_days=30)
        # Holdout indices should be the last ones
        assert holdout_idx[-1] == len(hourly_index_1y) - 1
        assert pool_idx[-1] < holdout_idx[0]

    def test_holdout_starts_at_midnight(self, hourly_index_1y):
        _, holdout_idx = carve_holdout(hourly_index_1y, holdout_days=90)
        holdout_start = hourly_index_1y[holdout_idx[0]]
        assert holdout_start.hour == 0

    def test_too_many_holdout_days(self, hourly_index_1y):
        with pytest.raises(ValueError, match="holdout_days"):
            carve_holdout(hourly_index_1y, holdout_days=400)
