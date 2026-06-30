"""Tests for modeling/metrics.py."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.metrics import (
    calculate_metrics,
    calculate_peak_metrics,
    calculate_pi_metrics,
)


class TestCalculateMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = calculate_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["me"] == 0.0
        assert m["r2"] == 1.0

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        m = calculate_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.5)
        assert m["me"] == pytest.approx(-1 / 6)  # mean of [-0.5, -0.5, 0.5]
        assert "rmse" in m
        assert "r2" in m
        assert "mape" in m
        assert "smape" in m

    def test_skill_scores(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])  # MAE=1
        y_baseline = np.array([15.0, 15.0, 35.0, 35.0])  # MAE=5
        m = calculate_metrics(y_true, y_pred, y_baseline=y_baseline)
        assert m["mae_skill"] == pytest.approx(0.8)  # 1 - 1/5
        assert "rmse_skill" in m

    def test_returns_all_keys(self):
        m = calculate_metrics([1, 2, 3], [1.1, 2.1, 3.1])
        expected_keys = {"mae", "rmse", "me", "r2", "mape", "smape"}
        assert expected_keys == set(m.keys())


class TestCalculatePiMetrics:
    def test_full_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        m = calculate_pi_metrics(y, y - 1, y + 1)
        assert m["pi_coverage"] == 1.0
        assert m["pi_mean_width"] == pytest.approx(2.0)

    def test_no_coverage(self):
        y = np.array([10.0, 20.0, 30.0])
        m = calculate_pi_metrics(y, np.zeros(3), np.ones(3))
        assert m["pi_coverage"] == 0.0

    def test_partial_coverage(self):
        y = np.array([1.0, 5.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])
        m = calculate_pi_metrics(y, lower, upper)
        assert m["pi_coverage"] == pytest.approx(0.5)


class TestCalculatePeakMetrics:
    def test_peak_hours_filter(self):
        # 48 hours of data
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        y_true = np.ones(48)
        y_pred = np.ones(48) + 0.5  # constant bias

        m = calculate_peak_metrics(y_true, y_pred, index=idx, peak_hours=[8, 9, 10])
        assert m["peak_mae"] == pytest.approx(0.5)
        assert m["peak_me"] == pytest.approx(-0.5)

    def test_requires_index(self):
        with pytest.raises(ValueError, match="index"):
            calculate_peak_metrics([1, 2], [1, 2])
