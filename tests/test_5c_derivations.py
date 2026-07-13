"""Tests for Stage 5c.0 feature infrastructure: EEG regime, neg_price stats,
gen/load forecast loader, and the engine branches that wire them together."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.config.features import GENERATION_COLUMNS
from energy_forecasting.features.engine import engineer_features
from energy_forecasting.features.market import (
    compute_eeg_regime,
    compute_generation_pct,
    compute_neg_price_stats,
)
from energy_forecasting.features.validation import validate_features

# ── compute_eeg_regime ──────────────────────────────────────────────


def test_eeg_regime_pre_2023_is_zero():
    idx = pd.date_range("2022-06-01", "2022-12-31 23:00", freq="h", tz="UTC")
    assert compute_eeg_regime(idx).eq(0).all()


@pytest.mark.parametrize(
    "ts,expected",
    [
        ("2022-12-31 23:00", 0),
        ("2023-01-01 00:00", 1),
        ("2023-12-31 23:00", 1),
        ("2024-01-01 00:00", 2),
        ("2025-02-24 23:00", 2),
        ("2025-02-25 00:00", 3),
        ("2026-04-01 12:00", 3),
    ],
)
def test_eeg_regime_transitions(ts, expected):
    idx = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")])
    assert compute_eeg_regime(idx).iloc[0] == expected


def test_eeg_regime_naive_index_supported():
    idx = pd.DatetimeIndex(["2022-12-31 23:00", "2023-01-01 00:00"])
    out = compute_eeg_regime(idx)
    assert out.tolist() == [0, 1]


# ── compute_neg_price_stats ─────────────────────────────────────────


def test_neg_price_stats_columns_and_dtype():
    idx = pd.date_range("2024-01-01", periods=24 * 100, freq="h", tz="UTC")
    price = pd.Series(np.full(len(idx), 50.0), index=idx)
    stats = compute_neg_price_stats(price)
    assert list(stats.columns) == [
        "_derived_neg_price_frac_30d",
        "_derived_neg_price_frac_90d",
        "_derived_neg_price_depth_30d",
    ]
    assert stats.index.equals(idx)


def test_neg_price_stats_zero_negatives():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    price = pd.Series(np.full(len(idx), 50.0), index=idx)
    stats = compute_neg_price_stats(price)
    assert stats["_derived_neg_price_frac_30d"].eq(0).all()
    assert stats["_derived_neg_price_frac_90d"].eq(0).all()
    assert stats["_derived_neg_price_depth_30d"].eq(0).all()


def test_neg_price_stats_all_negatives():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    price = pd.Series(np.full(len(idx), -10.0), index=idx)
    stats = compute_neg_price_stats(price)
    # After the rolling window fills (30d × 24h), frac = 1.0
    assert stats["_derived_neg_price_frac_30d"].iloc[-1] == 1.0
    assert stats["_derived_neg_price_frac_90d"].iloc[-1] == 1.0
    assert stats["_derived_neg_price_depth_30d"].iloc[-1] == 10.0


def test_neg_price_stats_known_fraction():
    # Every 10th hour is negative (-5), the rest at 40.
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    price = pd.Series(40.0, index=idx)
    price.iloc[::10] = -5.0
    stats = compute_neg_price_stats(price)
    # After 30 days the window is full → exactly 1/10 of hours are negative.
    tail = stats["_derived_neg_price_frac_30d"].iloc[-1]
    assert tail == pytest.approx(0.1, abs=1e-9)
    assert stats["_derived_neg_price_depth_30d"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def test_neg_price_stats_expand_before_full_window():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    price = pd.Series([-5.0, 10.0, -15.0, 20.0], index=idx)
    stats = compute_neg_price_stats(price)
    assert stats["_derived_neg_price_frac_30d"].tolist() == [1.0, 0.5, 2 / 3, 0.5]
    assert stats["_derived_neg_price_depth_30d"].tolist() == [5.0, 2.5, 20 / 3, 5.0]


# ── compute_generation_pct (per-technology forecast) ────────────────


def test_pct_forecast_per_technology_emits_three_columns():
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "forecast_gen_total": 60_000.0,
            "forecast_gen_solar": 12_000.0,
            "forecast_gen_wind_on": 18_000.0,
            "forecast_gen_wind_off": 6_000.0,
        },
        index=idx,
    )
    # Need generation columns to satisfy `compute_generation_pct`'s totals.
    for col in GENERATION_COLUMNS:
        df[col] = 1.0
    out = compute_generation_pct(df, add_forecast_pct=True)
    assert out["_derived_pct_forecast_solar"].iloc[0] == pytest.approx(12_000 / 60_000)
    assert out["_derived_pct_forecast_wind_on"].iloc[0] == pytest.approx(18_000 / 60_000)
    assert out["_derived_pct_forecast_wind_off"].iloc[0] == pytest.approx(6_000 / 60_000)


# ── Engine integration ─────────────────────────────────────────────


@pytest.fixture
def merged_like_df():
    idx = pd.date_range("2026-02-10", "2026-02-12 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    price = pd.Series(40.0 + rng.normal(0, 10, len(idx)), index=idx)
    price.iloc[::15] = -5.0
    df = pd.DataFrame(
        {
            "target_price": price,
            "forecast_gen_total": 60_000.0,
            "forecast_gen_solar": 8_000.0,
            "forecast_gen_wind_on": 18_000.0,
            "forecast_gen_wind_off": 6_000.0,
        },
        index=idx,
    )
    for col in GENERATION_COLUMNS:
        df[col] = 1_000.0
    return df


def test_engine_wires_eeg_neg_price_and_pct_forecast(merged_like_df):
    # Forecast artifacts are resolved before feature engineering. This test
    # covers eeg_regime, neg_price_*, and pct_forecast_*.
    feats = [
        "eeg_regime",
        "neg_price_frac_30d_d1",
        "neg_price_depth_30d_d1",
        "pct_forecast_solar",
        "pct_forecast_wind_on",
        "pct_forecast_wind_off",
    ]
    result = engineer_features(merged_like_df, feats)
    assert list(result.columns) == feats
    assert result["eeg_regime"].iloc[0] == 3
    # _d1 lag pushes the first 24 hours of rolling-stat outputs to NaN
    assert result["neg_price_frac_30d_d1"].iloc[:24].isna().all()
    assert result["neg_price_frac_30d_d1"].iloc[24:].notna().all()
    assert result["pct_forecast_solar"].iloc[0] == pytest.approx(8_000 / 60_000)


# ── Availability rules / validation ────────────────────────────────


def test_validation_accepts_new_5c_features():
    feats = [
        "eeg_regime",
        "neg_price_frac_30d_d1",
        "neg_price_frac_90d_d1",
        "neg_price_depth_30d_d1",
        "pct_forecast_solar",
        "pct_forecast_wind_on",
        "pct_forecast_wind_off",
    ]
    errors = validate_features(feats)
    assert errors == []


def test_validation_rejects_bare_neg_price():
    # neg_price_* maps to availability offset=-1, so a bare name must be flagged.
    errors = validate_features(["neg_price_frac_30d"])
    assert len(errors) == 1
    assert "Bare column name" in errors[0].reason
