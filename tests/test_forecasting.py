"""Tests for modeling/forecasting.py."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.forecasting import (
    _extract_lag,
    find_target_lag_columns,
    forecast_direct,
    forecast_with_lags,
    forecast_with_lags_windowed,
)
from sklearn.linear_model import Ridge


@pytest.fixture
def trained_model():
    """A simple fitted Ridge model."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 3))
    y_train = X_train[:, 0] * 2 + X_train[:, 1]
    model = Ridge()
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def test_data():
    """Test DataFrame with features."""
    idx = pd.date_range("2024-06-01", periods=48, freq="h")
    rng = np.random.default_rng(55)
    return pd.DataFrame(
        {f"f{i}": rng.normal(size=48) for i in range(3)},
        index=idx,
    )


class TestForecastDirect:
    def test_returns_dataframe(self, trained_model, test_data):
        result = forecast_direct(trained_model, test_data)
        assert isinstance(result, pd.DataFrame)
        assert "fitted" in result.columns
        assert "lower" in result.columns
        assert "upper" in result.columns
        assert len(result) == 48

    def test_no_intervals_gives_nan_bounds(self, trained_model, test_data):
        result = forecast_direct(trained_model, test_data)
        # Plain sklearn model has no predict_interval
        assert np.all(np.isnan(result["lower"].values))
        assert np.all(np.isnan(result["upper"].values))

    def test_preserves_index(self, trained_model, test_data):
        result = forecast_direct(trained_model, test_data)
        pd.testing.assert_index_equal(result.index, test_data.index)


class TestExtractLag:
    def test_simple_lag(self):
        assert _extract_lag("target_h24") == 24
        assert _extract_lag("price_h48") == 48
        assert _extract_lag("load_h168") == 168

    def test_target_lag_naming(self):
        # New target-lag naming (same `_h{N}` convention as TSO features)
        assert _extract_lag("wind_onshore_h1") == 1
        assert _extract_lag("wind_onshore_h12") == 12
        assert _extract_lag("wind_offshore_h6") == 6

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            _extract_lag("no_lag_here")


class TestFindTargetLagColumns:
    def test_matches_target_prefix(self):
        cols = [
            "wind_onshore_h1", "wind_onshore_h2", "wind_onshore_h12",
            "gen_wind_on_h24",  # TSO feature — different prefix
            "load_h24",  # TSO feature
            "hour_sin",
        ]
        assert find_target_lag_columns(cols, "wind_onshore") == [
            "wind_onshore_h1", "wind_onshore_h2", "wind_onshore_h12",
        ]

    def test_sorts_by_lag_hour(self):
        cols = ["load_h12", "load_h1", "load_h6"]
        assert find_target_lag_columns(cols, "load") == [
            "load_h1", "load_h6", "load_h12",
        ]

    def test_ignores_non_digit_suffix(self):
        cols = ["wind_onshore_hello", "wind_onshore_h", "wind_onshore_h7"]
        assert find_target_lag_columns(cols, "wind_onshore") == ["wind_onshore_h7"]

    def test_empty_when_no_match(self):
        assert find_target_lag_columns(["a", "b", "c"], "wind_onshore") == []

    def test_disambiguates_tso_features(self):
        # `load` target must not pick up `gen_wind_on_h24` etc.
        cols = ["load_h1", "load_h24", "gen_wind_on_h24", "gen_solar_h24"]
        assert find_target_lag_columns(cols, "load") == ["load_h1", "load_h24"]


class TestForecastWithLags:
    def test_overrides_leaky_lag_with_prediction(self):
        """A constant-prediction model must still propagate its prediction
        into future lag cells rather than letting the pre-filled actual
        leak through."""

        idx = pd.date_range("2024-06-01", periods=5, freq="h")
        # Pre-filled lag column contains a clearly distinguishable "leaky"
        # actual series. If recursive forecasting works, the model's constant
        # prediction (42.0) overwrites rows 1..4 before they're predicted.
        X_test = pd.DataFrame(
            {
                "f0": np.zeros(5),
                "target_h1": [7.0, 999.0, 999.0, 999.0, 999.0],
            },
            index=idx,
        )

        class ConstModel:
            def predict(self, X):
                return np.full(len(X), 42.0)

        y_train = pd.Series([1.0, 2.0, 3.0])
        result = forecast_with_lags(
            ConstModel(), X_test, y_train, ["target_h1"],
        )

        assert list(result["fitted"].values) == [42.0, 42.0, 42.0, 42.0, 42.0]
        assert list(result.index) == list(idx)

    def test_row_i_sees_previous_prediction_as_lag_1(self):
        """An identity-on-lag_1 model should echo the lag_1 value. The first
        row sees y_train's tail (pre-filled), subsequent rows should see the
        previous prediction."""

        idx = pd.date_range("2024-06-01", periods=4, freq="h")
        X_test = pd.DataFrame(
            {
                "target_h1": [7.0, 999.0, 999.0, 999.0],
            },
            index=idx,
        )

        class EchoLagModel:
            def predict(self, X):
                # X is a 1-row DataFrame; return its lag_1 value
                return X["target_h1"].values

        y_train = pd.Series([1.0, 2.0, 7.0])
        result = forecast_with_lags(
            EchoLagModel(), X_test, y_train, ["target_h1"],
        )
        # Row 0: lag_1 = 7.0 (from pre-fill), predicts 7.0
        # Row 1: lag_1 overwritten to 7.0, predicts 7.0
        # ... and so on — the echo stays at 7.0 throughout
        assert list(result["fitted"].values) == [7.0, 7.0, 7.0, 7.0]

    def test_empty_lag_columns_raises(self):
        idx = pd.date_range("2024-06-01", periods=3, freq="h")
        X_test = pd.DataFrame({"f0": np.zeros(3)}, index=idx)

        class ConstModel:
            def predict(self, X):
                return np.zeros(len(X))

        with pytest.raises(ValueError, match="at least one lag column"):
            forecast_with_lags(ConstModel(), X_test, pd.Series([1.0]), [])


class TestForecastWithLagsWindowed:
    @staticmethod
    def _make_const_model(value=42.0):
        class ConstModel:
            def predict(self, X):
                return np.full(len(X), value)

        return ConstModel()

    def test_full_coverage_with_none(self):
        """sample_windows=None covers every non-overlapping window."""
        idx = pd.date_range("2024-06-01", periods=10, freq="h")
        X = pd.DataFrame(
            {"target_h1": np.arange(10, dtype=float)}, index=idx,
        )
        y_pred, mask = forecast_with_lags_windowed(
            self._make_const_model(7.0), X, pd.Series([1.0, 2.0]),
            ["target_h1"], window_size=3, sample_windows=None,
        )
        # 10 rows / 3 window_size = 3 full windows, last row (index 9) dropped
        assert mask.sum() == 9
        assert np.all(y_pred[:9] == 7.0)
        assert np.isnan(y_pred[9])

    def test_single_sampled_window_takes_last(self):
        """sample_windows=1 picks the most recent window (most realistic)."""
        idx = pd.date_range("2024-06-01", periods=12, freq="h")
        X = pd.DataFrame(
            {"target_h1": np.arange(12, dtype=float)}, index=idx,
        )
        y_pred, mask = forecast_with_lags_windowed(
            self._make_const_model(9.0), X, pd.Series([1.0]),
            ["target_h1"], window_size=4, sample_windows=1,
        )
        # 12 rows / 4 = 3 possible windows; window 2 is the last (rows 8..11)
        assert mask.sum() == 4
        assert np.all(mask[8:12])
        assert not np.any(mask[:8])
        assert np.all(y_pred[8:12] == 9.0)

    def test_window_boundaries_reset_lag_seed(self):
        """Errors should not compound across windows: each new window
        re-seeds from the pre-filled (actual) lag column."""
        idx = pd.date_range("2024-06-01", periods=6, freq="h")
        # Pre-fill the lag column with distinct "actual" markers: each
        # window boundary row (0 and 3) holds a value that tells us the
        # seed came from actuals, not from the previous window.
        X = pd.DataFrame(
            {"target_h1": [100.0, 999.0, 999.0, 200.0, 999.0, 999.0]},
            index=idx,
        )

        class EchoLagModel:
            def predict(self, X):
                return X["target_h1"].values

        y_pred, mask = forecast_with_lags_windowed(
            EchoLagModel(), X, pd.Series([1.0]),
            ["target_h1"], window_size=3, sample_windows=None,
        )
        # Window 1 (rows 0..2): seed 100 — echo stays 100 throughout.
        # Window 2 (rows 3..5): seed 200 — fresh start, echo 200 throughout.
        assert list(y_pred) == [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]

    def test_returns_empty_mask_on_empty_frame(self):
        idx = pd.date_range("2024-06-01", periods=0, freq="h")
        X = pd.DataFrame({"target_h1": []}, index=idx)
        y_pred, mask = forecast_with_lags_windowed(
            self._make_const_model(), X, pd.Series([1.0]),
            ["target_h1"], window_size=3, sample_windows=None,
        )
        assert len(y_pred) == 0
        assert mask.sum() == 0

    def test_requires_lag_columns(self):
        idx = pd.date_range("2024-06-01", periods=3, freq="h")
        X = pd.DataFrame({"f": np.zeros(3)}, index=idx)
        with pytest.raises(ValueError, match="lag_columns"):
            forecast_with_lags_windowed(
                self._make_const_model(), X, pd.Series([1.0]),
                [], window_size=3,
            )
