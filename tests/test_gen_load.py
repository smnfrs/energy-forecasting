"""Tests for modeling/gen_load.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.gen_load import (
    _compute_temporal_features,
    _experiment_for_target,
    _get_target_col,
    _make_model,
)
from energy_forecasting.modeling.training import _apply_scaler, _fit_scaler

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def hourly_index_90d():
    """90 days of hourly UTC data."""
    return pd.date_range("2024-01-01", periods=90 * 24, freq="h", tz="UTC")


@pytest.fixture
def tso_df(hourly_index_90d):
    """Synthetic TSO DataFrame mimicking 50Hertz structure."""
    rng = np.random.default_rng(42)
    n = len(hourly_index_90d)
    return pd.DataFrame(
        {
            "wind_onshore_50hz": rng.uniform(100, 5000, n),
            "wind_offshore_50hz": rng.uniform(50, 2000, n),
            "solar_50hz": rng.uniform(0, 3000, n),
            "load_50hz": rng.uniform(3000, 8000, n),
        },
        index=hourly_index_90d,
    )


# ── TestComputeTemporalFeatures ──────────────────────────────────


class TestComputeTemporalFeatures:
    def test_produces_cyclical_features(self, tso_df):
        result = _compute_temporal_features(tso_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_produces_calendar_features(self, tso_df):
        result = _compute_temporal_features(tso_df)
        assert "is_weekend" in result.columns
        assert "is_holiday" in result.columns
        assert "day_index" in result.columns

    def test_produces_lagged_features(self, tso_df):
        result = _compute_temporal_features(tso_df)
        # Should have h24 lag and d7_d2_avg for each gen type
        assert "gen_wind_on_h24" in result.columns
        assert "load_h24" in result.columns
        assert "gen_wind_on_d7_d2_avg" in result.columns

    def test_lag_values_correct(self, tso_df):
        result = _compute_temporal_features(tso_df)
        # h24 lag: value at index 24 should equal original at index 0
        expected = tso_df["wind_onshore_50hz"].iloc[0]
        actual = result["gen_wind_on_h24"].iloc[24]
        assert actual == pytest.approx(expected)

    def test_first_24_rows_nan_for_lags(self, tso_df):
        result = _compute_temporal_features(tso_df)
        assert result["gen_wind_on_h24"].iloc[:24].isna().all()

    def test_preserves_index(self, tso_df):
        result = _compute_temporal_features(tso_df)
        pd.testing.assert_index_equal(result.index, tso_df.index)

    def test_fourier_features_present(self, tso_df):
        result = _compute_temporal_features(tso_df)
        fourier_cols = [c for c in result.columns if "fourier" in c]
        assert len(fourier_cols) >= 1


# ── TestGetTargetCol ─────────────────────────────────────────────


class TestGetTargetCol:
    def test_50hz_wind_onshore(self):
        assert _get_target_col("wind_onshore", "DE_50HZ") == "wind_onshore_50hz"

    def test_amprion_load(self):
        assert _get_target_col("load", "DE_AMPRION") == "load_ampr"

    def test_tennet_solar(self):
        assert _get_target_col("solar", "DE_TENNET") == "solar_tenn"

    def test_gen_load_diff_national(self):
        assert _get_target_col("gen_load_diff", "DE_NATIONAL") == "gen_load_diff"


# ── TestMakeModel ────────────────────────────────────────────────


class TestMakeModel:
    def test_lgbm(self):
        model = _make_model("LGBMRegressor", {
            "n_estimators": 100, "learning_rate": 0.1, "objective": "mae",
            "metric": "mae",
        })
        assert type(model).__name__ == "LGBMRegressor"

    def test_xgboost(self):
        model = _make_model("XGBRegressor", {
            "n_estimators": 100, "learning_rate": 0.1,
            "objective": "reg:absoluteerror", "eval_metric": "mae",
        })
        assert type(model).__name__ == "XGBRegressor"

    def test_elasticnet(self):
        model = _make_model("ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5})
        assert type(model).__name__ == "ElasticNet"

    def test_ridge(self):
        model = _make_model("Ridge", {"alpha": 1.0})
        assert type(model).__name__ == "Ridge"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            _make_model("GradientBoosting", {})

    def test_lgbm_fit_predict(self):
        model = _make_model("LGBMRegressor", {
            "n_estimators": 10, "learning_rate": 0.1, "objective": "mae",
            "metric": "mae", "num_leaves": 8,
        })
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 100)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)


# ── TestScaler ───────────────────────────────────────────────────


class TestScaler:
    def test_fit_standard(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scaler = _fit_scaler("standard", X)
        assert scaler is not None
        result = _apply_scaler(scaler, X)
        assert result.shape == (3, 2)
        # StandardScaler: mean ~0
        assert np.abs(result.mean()) < 0.01

    def test_fit_none(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        scaler = _fit_scaler("none", X)
        assert scaler is None
        result = _apply_scaler(scaler, X)
        np.testing.assert_array_equal(result, [[1], [2], [3]])

    def test_fit_minmax(self):
        X = pd.DataFrame({"a": [0, 5, 10]})
        scaler = _fit_scaler("minmax", X)
        result = _apply_scaler(scaler, X)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)


# ── TestExperimentForTarget ──────────────────────────────────────


class TestExperimentForTarget:
    def test_wind_onshore(self):
        assert _experiment_for_target("wind_onshore") == "gen_wind_onshore"

    def test_load(self):
        assert _experiment_for_target("load") == "gen_load"

    def test_gen_load_diff(self):
        assert _experiment_for_target("gen_load_diff") == "gen_gen_load_diff"

    def test_solar(self):
        assert _experiment_for_target("solar") == "gen_solar"


# ── TestBuildFeaturesWithExog ────────────────────────────────────


class TestExogConfig:
    def test_load_has_exog_targets(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS

        assert GEN_LOAD_TARGETS["load"]["exog_targets"] == [
            "wind_onshore", "wind_offshore", "solar",
        ]

    def test_gen_load_diff_has_exog_targets(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS

        assert "load" in GEN_LOAD_TARGETS["gen_load_diff"]["exog_targets"]
        assert "wind_onshore" in GEN_LOAD_TARGETS["gen_load_diff"]["exog_targets"]

    def test_wind_solar_have_no_exog(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS

        for target in ["wind_onshore", "wind_offshore", "solar"]:
            assert GEN_LOAD_TARGETS[target]["exog_targets"] == []


# ── TestTrainingOrder ────────────────────────────────────────────


class TestTrainingOrder:
    def test_order_has_all_targets(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS, GEN_LOAD_TRAINING_ORDER

        all_ordered = [t for group in GEN_LOAD_TRAINING_ORDER for t in group]
        assert set(all_ordered) == set(GEN_LOAD_TARGETS.keys())

    def test_independent_targets_first(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TARGETS, GEN_LOAD_TRAINING_ORDER

        # First group should have no exog dependencies
        for t in GEN_LOAD_TRAINING_ORDER[0]:
            assert GEN_LOAD_TARGETS[t]["exog_targets"] == []

    def test_load_after_generation(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TRAINING_ORDER

        flat = [t for group in GEN_LOAD_TRAINING_ORDER for t in group]
        load_idx = flat.index("load")
        for gen_target in ["wind_onshore", "wind_offshore", "solar"]:
            assert flat.index(gen_target) < load_idx

    def test_gen_load_diff_last(self):
        from energy_forecasting.config.modeling import GEN_LOAD_TRAINING_ORDER

        flat = [t for group in GEN_LOAD_TRAINING_ORDER for t in group]
        assert flat[-1] == "gen_load_diff"


# ── TestSuggestDatasetParams ──────────────────────────────────────


class TestSuggestDatasetParams:
    def test_elasticnet_pins_log_target_false(self):
        import optuna
        from energy_forecasting.config.search_spaces import suggest_dataset_params

        study = optuna.create_study()
        for _ in range(20):
            trial = study.ask()
            params = suggest_dataset_params(trial, model_type="ElasticNet")
            assert params["log_target"] is False

    def test_other_models_can_pick_log_target_true(self):
        import optuna
        from energy_forecasting.config.search_spaces import suggest_dataset_params

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        observed = set()
        for _ in range(50):
            trial = study.ask()
            params = suggest_dataset_params(trial, model_type="LGBMRegressor")
            observed.add(params["log_target"])
        assert observed == {True, False}

    def test_default_call_unrestricted(self):
        import optuna
        from energy_forecasting.config.search_spaces import suggest_dataset_params

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        observed = set()
        for _ in range(50):
            trial = study.ask()
            params = suggest_dataset_params(trial)
            observed.add(params["log_target"])
        assert observed == {True, False}
