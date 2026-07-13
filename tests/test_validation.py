"""Tests for features/validation.py — leakage validation."""

import pytest
from energy_forecasting.features.validation import validate_features, validate_price_feature_list

# ── Valid features (should produce 0 errors) ──────────────────────


class TestValidFeatures:
    """Features that should pass validation."""

    @pytest.mark.parametrize(
        "feature",
        [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "is_weekend",
            "is_holiday",
            "day_index",
        ],
    )
    def test_deterministic_always_valid(self, feature):
        assert validate_features([feature]) == []

    @pytest.mark.parametrize(
        "feature",
        [
            "forecast_gen_total_daily_sum",
            "forecast_gen_wind_pv_daily_max",
            "forecast_load_daily_sum",
        ],
    )
    def test_forecasts_daily_agg_valid(self, feature):
        assert validate_features([feature]) == []

    @pytest.mark.parametrize(
        "feature",
        [
            "price_h24",
            "price_h168",
            "price_d1",
            "price_d7_d1_std",
            "price_d14_d1_avg",
            "price_d7_d1_h8_h20_avg",
        ],
    )
    def test_price_features_valid(self, feature):
        assert validate_features([feature]) == []

    @pytest.mark.parametrize(
        "feature",
        [
            "gen_wind_on_d1_eh7",
            "gen_solar_d1_eh7",
            "gen_wind_on_d7_d2_avg",
            "residual_load_d1_eh7",
        ],
    )
    def test_generation_with_cutoff_valid(self, feature):
        assert validate_features([feature]) == []

    @pytest.mark.parametrize(
        "feature",
        [
            "ttf_d2",
            "brent_d2",
            "carbon_d2",
            "ttf_ewma_720_d2",
            "ttf_d7_d2_avg",
        ],
    )
    def test_commodity_features_valid(self, feature):
        assert validate_features([feature]) == []

    def test_ewma_with_cutoff_valid(self):
        assert validate_features(["price_ewma_6_d1"]) == []
        assert validate_features(["gen_wind_on_ewma_24_d1_h7"]) == []

    def test_interaction_valid(self):
        assert validate_features(["gen_wind_on_d1_eh7__x__day_index"]) == []


# ── Invalid features (should produce errors) ─────────────────────


class TestInvalidFeatures:
    """Features that should fail validation."""

    def test_bare_price_leaks(self):
        errors = validate_features(["price"])
        assert len(errors) == 1
        assert "Bare column name" in errors[0].reason

    def test_bare_generation_leaks(self):
        errors = validate_features(["gen_wind_on"])
        assert len(errors) == 1
        assert "Bare column name" in errors[0].reason

    def test_bare_commodity_leaks(self):
        errors = validate_features(["ttf"])
        assert len(errors) == 1

    def test_insufficient_hourly_lag(self):
        """price needs >=24h lag (offset=-1)."""
        errors = validate_features(["price_h12"])
        assert len(errors) == 1
        assert "insufficient" in errors[0].reason.lower()

    def test_gen_insufficient_hourly_lag(self):
        """gen_wind_on has offset=-1, cutoff=7, so needs >= 17h lag."""
        errors = validate_features(["gen_wind_on_h10"])
        assert len(errors) == 1

    def test_commodity_too_recent_end_day(self):
        """ttf has offset=-2, so _d1 is too recent."""
        errors = validate_features(["ttf_d1"])
        assert len(errors) == 1
        assert "too recent" in errors[0].reason.lower()

    def test_gen_full_day_without_cutoff(self):
        """gen_wind_on available only to hour 7, so _d1 without _eh is bad."""
        errors = validate_features(["gen_wind_on_d1"])
        assert len(errors) == 1
        assert "only available up to hour 7" in errors[0].reason

    def test_gen_end_hour_exceeds_cutoff(self):
        """end_hour=12 exceeds cutoff_hour=7."""
        errors = validate_features(["gen_wind_on_d1_eh12"])
        assert len(errors) == 1
        assert "exceeds cutoff_hour" in errors[0].reason

    def test_gen_eh10_rejected_after_cutoff_fix(self):
        """After the cutoff fix (stage 5c prerequisite), eh10 exceeds the new cutoff_hour=7.

        Inference runs at 08:00 UTC; D-1 data is only usable up to 07:00 UTC, so
        single-day D-1 features must use `_eh7` or earlier. Regression guard for
        the `_eh10` → `_eh7` fix.
        """
        errors = validate_features(["gen_wind_on_d1_eh10"])
        assert len(errors) == 1
        assert "exceeds cutoff_hour" in errors[0].reason

    def test_ewma_h10_rejected_after_cutoff_fix(self):
        """EWMA cutoff-hour on D-1 generation must also respect cutoff_hour=7."""
        errors = validate_features(["gen_wind_on_ewma_24_d1_h10"])
        assert len(errors) == 1

    def test_ewma_without_cutoff_day(self):
        """EWMA on non-deterministic data needs cutoff_day."""
        errors = validate_features(["price_ewma_6"])
        assert len(errors) == 1
        assert "cutoff_day" in errors[0].reason

    def test_ewma_too_recent_cutoff(self):
        """ttf has offset=-2, so cutoff_day=-1 is too recent."""
        errors = validate_features(["ttf_ewma_24_d1"])
        assert len(errors) == 1
        assert "too recent" in errors[0].reason.lower()

    def test_daily_agg_on_non_deterministic(self):
        """_daily_avg on generation data uses today's data — leaks."""
        errors = validate_features(["gen_wind_on_daily_avg"])
        assert len(errors) == 1
        assert "current-day" in errors[0].reason.lower()

    def test_interaction_left_leaks(self):
        """Left side of interaction is bare price — leaks."""
        errors = validate_features(["price__x__day_index"])
        assert len(errors) >= 1

    def test_multiple_errors_returned(self):
        """Multiple bad features produce multiple errors."""
        errors = validate_features(["price", "gen_wind_on", "ttf"])
        assert len(errors) == 3


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_gen_d2_no_cutoff_issue(self):
        """D-2 is far enough back that cutoff doesn't matter."""
        assert validate_features(["gen_wind_on_d7_d2_avg"]) == []

    def test_price_h24_exact_boundary(self):
        """h24 is exactly the minimum for offset=-1."""
        assert validate_features(["price_h24"]) == []

    def test_price_h23_too_short(self):
        """h23 is 1 hour short."""
        errors = validate_features(["price_h23"])
        assert len(errors) == 1

    def test_empty_feature_list(self):
        assert validate_features([]) == []

    def test_full_slim_list_valid(self):
        """The entire PRICE_FEATURES_SLIM list should validate."""
        from energy_forecasting.config.features import PRICE_FEATURES_SLIM

        errors = validate_features(PRICE_FEATURES_SLIM)
        assert errors == [], f"Errors: {[e.reason for e in errors]}"

    def test_full_full_list_valid(self):
        from energy_forecasting.config.features import PRICE_FEATURES_FULL

        errors = validate_features(PRICE_FEATURES_FULL)
        assert errors == [], f"Errors: {[e.reason for e in errors]}"


def test_validate_price_feature_list_rejects_raw_forecast_tokens():
    with pytest.raises(ValueError, match="source-neutral forecast"):
        validate_price_feature_list(["prog_residual__x__day_index"])


def test_availability_rejects_prog_feature():
    errors = validate_features(["prog_residual"])
    assert errors
    assert "No availability rule" in errors[0].reason


def test_price_feature_lists_contain_no_raw_forecast_tokens():
    from energy_forecasting.config.features import (
        PRICE_FEATURES_FULL,
        PRICE_FEATURES_MAX,
        PRICE_FEATURES_SLIM,
    )

    for feature_list in (PRICE_FEATURES_SLIM, PRICE_FEATURES_FULL, PRICE_FEATURES_MAX):
        validate_price_feature_list(feature_list)
