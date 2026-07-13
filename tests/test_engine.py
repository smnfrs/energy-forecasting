"""Tests for features/engine.py — feature computation engine."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.features.engine import engineer_features, extend_features


@pytest.fixture
def sample_df():
    """Minimal DataFrame that supports a range of feature computations."""
    idx = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")  # 10 days
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "target_price": rng.normal(50, 20, len(idx)),
            "marktpreis_frankreich": rng.normal(45, 18, len(idx)),
            "stromerzeugung_wind_onshore": rng.uniform(1000, 20000, len(idx)),
            "stromerzeugung_photovoltaik": rng.uniform(0, 15000, len(idx)),
            "forecast_gen_total": rng.uniform(30000, 70000, len(idx)),
            "forecast_gen_wind_pv": rng.uniform(5000, 30000, len(idx)),
        },
        index=idx,
    )


# ── Basic feature computation ─────────────────────────────────────


def test_temporal_features(sample_df):
    features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    result = engineer_features(sample_df, features, validate=True)
    assert set(features).issubset(result.columns)
    assert result.shape[0] == len(sample_df)
    assert result["hour_sin"].between(-1, 1).all()


def test_hourly_lag(sample_df):
    result = engineer_features(sample_df, ["price_h24"], validate=True)
    assert "price_h24" in result.columns
    assert pd.isna(result["price_h24"].iloc[0])
    assert result["price_h24"].iloc[24] == sample_df["target_price"].iloc[0]


def test_rolling_stat(sample_df):
    result = engineer_features(sample_df, ["price_d7_d1_std"], validate=True)
    assert "price_d7_d1_std" in result.columns
    # Should have NaN for first 7 days, then valid values
    last_day = result.loc["2024-01-10"]
    assert last_day["price_d7_d1_std"].notna().all()


def test_ewma_feature(sample_df):
    result = engineer_features(sample_df, ["price_ewma_6_d1"], validate=True)
    assert "price_ewma_6_d1" in result.columns


def test_fourier_features(sample_df):
    result = engineer_features(sample_df, ["hour_fourier_24_3"], validate=True)
    # 3 sin + 3 cos = 6 columns
    fourier_cols = [c for c in result.columns if "fourier" in c]
    assert len(fourier_cols) == 6


def test_daily_aggregate(sample_df):
    result = engineer_features(sample_df, ["forecast_gen_total_daily_sum"], validate=True)
    assert "forecast_gen_total_daily_sum" in result.columns


def test_interaction(sample_df):
    result = engineer_features(
        sample_df,
        ["gen_wind_on_d7_d2_avg__x__day_index"],
        validate=True,
    )
    assert "gen_wind_on_d7_d2_avg__x__day_index" in result.columns


# ── Validation integration ────────────────────────────────────────


def test_validation_rejects_bare_names(sample_df):
    with pytest.raises(ValueError, match="leakage"):
        engineer_features(sample_df, ["price"], validate=True)


def test_validation_can_be_disabled(sample_df):
    # Should not raise even with bare name
    result = engineer_features(sample_df, ["price"], validate=False)
    assert "price" in result.columns


# ── extend_features ───────────────────────────────────────────────


def test_extend_features(sample_df):
    """extend_features should add new rows without recomputing old ones."""
    features = ["price_h24", "hour_sin"]

    # Split: first 7 days for existing, full for extension
    cutoff = pd.Timestamp("2024-01-08", tz="UTC")
    df_initial = sample_df.loc[:cutoff]
    existing = engineer_features(df_initial, features, validate=True)

    extended = extend_features(existing, sample_df, features)
    assert len(extended) > len(existing)
    assert extended.index[-1] == sample_df.index[-1]


def test_extend_features_no_new_data(sample_df):
    features = ["hour_sin"]
    existing = engineer_features(sample_df, features, validate=True)
    extended = extend_features(existing, sample_df, features)
    assert len(extended) == len(existing)


# ── Missing column error ──────────────────────────────────────────


def test_missing_column_raises(sample_df):
    """Feature requiring a column not in df should raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        engineer_features(sample_df, ["brent_d2"], validate=False)
