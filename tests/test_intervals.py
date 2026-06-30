"""Tests for modeling/intervals.py."""

import numpy as np
import pytest
from energy_forecasting.modeling.intervals import (
    calibrate_ensemble_intervals,
    predict_ensemble_intervals,
    predict_with_intervals,
    wrap_with_intervals,
)
from sklearn.linear_model import Ridge


@pytest.fixture
def simple_data():
    """Simple regression data for interval testing."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.normal(0, 0.1, n)
    return X, y


class TestWrapWithIntervals:
    def test_creates_cross_conformal(self):
        from mapie.regression import CrossConformalRegressor

        model = wrap_with_intervals(Ridge())
        assert isinstance(model, CrossConformalRegressor)

    def test_custom_confidence(self):
        model = wrap_with_intervals(Ridge(), confidence_level=0.95)
        # MAPIE 1.3 stores confidence_level as _alphas = [1 - confidence]
        assert model._alphas == [pytest.approx(0.05)]


class TestPredictWithIntervals:
    def test_returns_three_arrays(self, simple_data):
        X, y = simple_data
        X_train, X_test = X[:400], X[400:]
        y_train = y[:400]

        model = wrap_with_intervals(Ridge(), cv=3)
        model.fit_conformalize(X_train, y_train)
        y_pred, lower, upper = predict_with_intervals(model, X_test)

        assert y_pred.shape == (100,)
        assert lower.shape == (100,)
        assert upper.shape == (100,)
        # Lower should be below upper
        assert np.all(lower <= upper)
        # Point predictions should be between bounds
        assert np.all(y_pred >= lower - 1e-6)
        assert np.all(y_pred <= upper + 1e-6)

    def test_coverage_approximate(self, simple_data):
        X, y = simple_data
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = wrap_with_intervals(Ridge(), confidence_level=0.90, cv=3)
        model.fit_conformalize(X_train, y_train)
        _, lower, upper = predict_with_intervals(model, X_test)

        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        # Should be approximately 90% ± some tolerance
        assert coverage >= 0.75, f"Coverage too low: {coverage}"


class TestEnsembleIntervals:
    def test_calibrate_quantile(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], dtype=float)
        q = calibrate_ensemble_intervals(y_true, y_pred, confidence_level=0.90)
        assert q > 0
        # All residuals are 0.5, so quantile should be ~0.5
        assert q == pytest.approx(0.5, abs=0.1)

    def test_predict_intervals_symmetric(self):
        y_pred = np.array([10.0, 20.0, 30.0])
        lower, upper = predict_ensemble_intervals(y_pred, conformal_quantile=5.0)
        np.testing.assert_array_equal(lower, [5.0, 15.0, 25.0])
        np.testing.assert_array_equal(upper, [15.0, 25.0, 35.0])

    def test_calibrate_then_predict_coverage(self):
        rng = np.random.default_rng(123)
        y_true = rng.normal(0, 1, 1000)
        y_pred = y_true + rng.normal(0, 0.5, 1000)

        q = calibrate_ensemble_intervals(y_true, y_pred, confidence_level=0.90)
        lower, upper = predict_ensemble_intervals(y_pred, q)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        assert coverage >= 0.85
