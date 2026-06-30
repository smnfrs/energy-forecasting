"""Tests for modeling/training.py."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.training import (
    build_pipeline,
    compute_sample_weights,
)
from sklearn.linear_model import Ridge


class TestBuildPipeline:
    def test_standard_scaler(self):
        pipe = build_pipeline(Ridge(), scaler="standard", target_transform="none")
        assert "scaler" in dict(pipe.steps)
        assert "model" in dict(pipe.steps)

    def test_robust_scaler(self):
        pipe = build_pipeline(Ridge(), scaler="robust")
        scaler_step = dict(pipe.steps)["scaler"]
        assert type(scaler_step).__name__ == "RobustScaler"

    def test_no_scaler(self):
        pipe = build_pipeline(Ridge(), scaler="none")
        step_names = [name for name, _ in pipe.steps]
        assert "scaler" not in step_names
        assert "model" in step_names

    def test_log_shift_transform(self):
        from sklearn.compose import TransformedTargetRegressor

        pipe = build_pipeline(Ridge(), target_transform="log_shift")
        model_step = dict(pipe.steps)["model"]
        assert isinstance(model_step, TransformedTargetRegressor)

    def test_yeo_johnson_transform(self):
        from sklearn.compose import TransformedTargetRegressor

        pipe = build_pipeline(Ridge(), target_transform="yeo_johnson")
        model_step = dict(pipe.steps)["model"]
        assert isinstance(model_step, TransformedTargetRegressor)

    def test_invalid_transform(self):
        with pytest.raises(ValueError, match="Unknown target_transform"):
            build_pipeline(Ridge(), target_transform="invalid")

    def test_fit_predict(self):
        """Pipeline should be fittable and predictable."""
        pipe = build_pipeline(Ridge(), scaler="standard", target_transform="log_shift")
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = np.abs(rng.normal(50, 10, 100))  # positive for log
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)
        # Predictions should be in a reasonable range
        assert np.all(np.isfinite(preds))


class TestComputeSampleWeights:
    def test_most_recent_weight_is_one(self):
        days = pd.Series([0.0, 100.0, 200.0, 365.0])
        weights = compute_sample_weights(days, half_life_days=365.0)
        assert weights[-1] == pytest.approx(1.0)

    def test_half_life(self):
        days = pd.Series([0.0, 365.0])
        weights = compute_sample_weights(days, half_life_days=365.0)
        # At half_life before the most recent, weight should be 0.5
        assert weights[0] == pytest.approx(0.5)

    def test_weights_monotonically_increase(self):
        days = pd.Series(np.arange(0, 730, 1, dtype=float))
        weights = compute_sample_weights(days, half_life_days=365.0)
        assert np.all(np.diff(weights) >= 0)

    def test_all_weights_positive(self):
        days = pd.Series(np.arange(0, 1000, 1, dtype=float))
        weights = compute_sample_weights(days, half_life_days=365.0)
        assert np.all(weights > 0)
