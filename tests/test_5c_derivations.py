"""Tests for Stage 5c.0 feature infrastructure: EEG regime, neg_price stats,
gen/load forecast loader, and the engine branches that wire them together."""

from __future__ import annotations

from pathlib import Path

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
from energy_forecasting.modeling.gen_load_forecasts import (
    FORECAST_TARGETS,
    load_gen_load_forecasts,
)

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


# ── compute_generation_pct (per-technology prognosis) ──────────────


def test_pct_prog_per_technology_emits_three_columns():
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "prognostizierte_erzeugung_gesamt": 60_000.0,
            "prognostizierte_erzeugung_photovoltaik": 12_000.0,
            "prognostizierte_erzeugung_onshore": 18_000.0,
            "prognostizierte_erzeugung_offshore": 6_000.0,
        },
        index=idx,
    )
    # Need generation columns to satisfy `compute_generation_pct`'s totals.
    for col in GENERATION_COLUMNS:
        df[col] = 1.0
    out = compute_generation_pct(df, add_prognosticated_pct=True)
    assert out["_derived_pct_prog_solar"].iloc[0] == pytest.approx(12_000 / 60_000)
    assert out["_derived_pct_prog_wind_on"].iloc[0] == pytest.approx(18_000 / 60_000)
    assert out["_derived_pct_prog_wind_off"].iloc[0] == pytest.approx(6_000 / 60_000)


# ── load_gen_load_forecasts ─────────────────────────────────────────


def _write_forecast_parquet(path: Path, idx: pd.DatetimeIndex, value: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y_true": np.full(len(idx), value),
            "y_pred": np.full(len(idx), value),
            "y_lower": np.nan,
            "y_upper": np.nan,
        },
        index=idx,
    ).to_parquet(path)


@pytest.fixture
def fake_forecasts_root(tmp_path):
    idx = pd.date_range("2026-02-01", "2026-02-28 23:00", freq="h", tz="UTC")
    values = {
        "wind_onshore": 10_000.0,
        "wind_offshore": 4_000.0,
        "solar": 2_000.0,
        "load": 50_000.0,
        "gen_load_diff": -1_000.0,
    }
    for target, value in values.items():
        _write_forecast_parquet(tmp_path / f"{target}_DE_NATIONAL.parquet", idx, value)
    return tmp_path


def test_load_gen_load_forecasts_returns_all_columns(fake_forecasts_root):
    idx = pd.date_range("2026-02-05", "2026-02-07 23:00", freq="h", tz="UTC")
    out = load_gen_load_forecasts(idx, root=fake_forecasts_root)
    assert list(out.columns) == [
        "_derived_forecast_wind_on",
        "_derived_forecast_wind_off",
        "_derived_forecast_solar",
        "_derived_forecast_load",
        "_derived_forecast_gen_load_diff",
        "_derived_forecast_residual",
    ]
    assert out.index.equals(idx)
    # residual = load - wind_on - wind_off - solar = 50000 - 10000 - 4000 - 2000 = 34000
    assert out["_derived_forecast_residual"].iloc[0] == pytest.approx(34_000.0)


def test_load_gen_load_forecasts_subset(fake_forecasts_root):
    idx = pd.date_range("2026-02-10", "2026-02-10 03:00", freq="h", tz="UTC")
    out = load_gen_load_forecasts(
        idx,
        columns=["_derived_forecast_load", "_derived_forecast_solar"],
        root=fake_forecasts_root,
    )
    assert list(out.columns) == ["_derived_forecast_load", "_derived_forecast_solar"]


def test_load_gen_load_forecasts_residual_pulls_inputs(fake_forecasts_root):
    idx = pd.date_range("2026-02-10", "2026-02-10 03:00", freq="h", tz="UTC")
    out = load_gen_load_forecasts(
        idx, columns=["_derived_forecast_residual"], root=fake_forecasts_root
    )
    # The function returns only what was requested; residual must equal
    # load - wind_on - wind_off - solar (50000 - 10000 - 4000 - 2000).
    assert list(out.columns) == ["_derived_forecast_residual"]
    assert out.iloc[0, 0] == pytest.approx(34_000.0)


def test_load_gen_load_forecasts_missing_file_raises(tmp_path):
    idx = pd.date_range("2026-02-01", "2026-02-01 03:00", freq="h", tz="UTC")
    with pytest.raises(FileNotFoundError, match="missing historical_forecasts parquet"):
        load_gen_load_forecasts(idx, root=tmp_path)


def test_load_gen_load_forecasts_unknown_column_raises(fake_forecasts_root):
    idx = pd.date_range("2026-02-01", "2026-02-01 03:00", freq="h", tz="UTC")
    with pytest.raises(ValueError, match="Unknown gen/load forecast column"):
        load_gen_load_forecasts(idx, columns=["_derived_forecast_nope"], root=fake_forecasts_root)


def test_load_gen_load_forecasts_handles_naive_index(fake_forecasts_root):
    # Merged dataset uses tz-naive UTC; loader must align to source's tz.
    idx = pd.date_range("2026-02-05", "2026-02-05 03:00", freq="h")
    out = load_gen_load_forecasts(idx, root=fake_forecasts_root)
    assert out.index.tz is None
    assert out["_derived_forecast_load"].notna().all()


def test_load_gen_load_forecasts_converts_utc_to_naive_berlin_winter(tmp_path):
    src_idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01 00:00", tz="UTC")])
    _write_forecast_parquet(tmp_path / "load_DE_NATIONAL.parquet", src_idx, 123.0)
    target_idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01 01:00")])
    out = load_gen_load_forecasts(target_idx, columns=["_derived_forecast_load"], root=tmp_path)
    assert out["_derived_forecast_load"].iloc[0] == pytest.approx(123.0)


def test_load_gen_load_forecasts_converts_utc_to_naive_berlin_summer(tmp_path):
    src_idx = pd.DatetimeIndex([pd.Timestamp("2026-07-01 00:00", tz="UTC")])
    _write_forecast_parquet(tmp_path / "load_DE_NATIONAL.parquet", src_idx, 456.0)
    target_idx = pd.DatetimeIndex([pd.Timestamp("2026-07-01 02:00")])
    out = load_gen_load_forecasts(target_idx, columns=["_derived_forecast_load"], root=tmp_path)
    assert out["_derived_forecast_load"].iloc[0] == pytest.approx(456.0)


def test_load_gen_load_forecasts_averages_dst_fallback_duplicate(tmp_path):
    path = tmp_path / "load_DE_NATIONAL.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    src_idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-10-25 00:00", tz="UTC"),
            pd.Timestamp("2026-10-25 01:00", tz="UTC"),
        ]
    )
    pd.DataFrame(
        {"y_true": [10.0, 30.0], "y_pred": [10.0, 30.0], "y_lower": np.nan, "y_upper": np.nan},
        index=src_idx,
    ).to_parquet(path)
    target_idx = pd.DatetimeIndex([pd.Timestamp("2026-10-25 02:00")])
    out = load_gen_load_forecasts(target_idx, columns=["_derived_forecast_load"], root=tmp_path)
    assert out["_derived_forecast_load"].iloc[0] == pytest.approx(20.0)


def test_forecast_targets_cover_documented_short_names():
    # Sanity that the loader knows about each forecast short name from columns.py
    expected = {
        "_derived_forecast_wind_on",
        "_derived_forecast_wind_off",
        "_derived_forecast_solar",
        "_derived_forecast_load",
        "_derived_forecast_gen_load_diff",
    }
    assert set(FORECAST_TARGETS.keys()) == expected


# ── Engine integration ─────────────────────────────────────────────


@pytest.fixture
def merged_like_df(fake_forecasts_root):
    idx = pd.date_range("2026-02-10", "2026-02-12 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    price = pd.Series(40.0 + rng.normal(0, 10, len(idx)), index=idx)
    price.iloc[::15] = -5.0
    df = pd.DataFrame(
        {
            "target_price": price,
            "prognostizierte_erzeugung_gesamt": 60_000.0,
            "prognostizierte_erzeugung_photovoltaik": 8_000.0,
            "prognostizierte_erzeugung_onshore": 18_000.0,
            "prognostizierte_erzeugung_offshore": 6_000.0,
        },
        index=idx,
    )
    for col in GENERATION_COLUMNS:
        df[col] = 1_000.0
    df.attrs["forecasts_root"] = fake_forecasts_root
    return df


def test_engine_wires_eeg_neg_price_and_pct_prog(merged_like_df):
    # The EMA-historical forecast overlay no longer flows through the
    # feature engine — it's applied at price-dataset prep onto prog_*
    # columns. This test now covers eeg_regime, neg_price_*, pct_prog_*.
    feats = [
        "eeg_regime",
        "neg_price_frac_30d_d1",
        "neg_price_depth_30d_d1",
        "pct_prog_solar",
        "pct_prog_wind_on",
        "pct_prog_wind_off",
    ]
    result = engineer_features(merged_like_df, feats)
    assert list(result.columns) == feats
    assert result["eeg_regime"].iloc[0] == 3
    # _d1 lag pushes the first 24 hours of rolling-stat outputs to NaN
    assert result["neg_price_frac_30d_d1"].iloc[:24].isna().all()
    assert result["neg_price_frac_30d_d1"].iloc[24:].notna().all()
    assert result["pct_prog_solar"].iloc[0] == pytest.approx(8_000 / 60_000)


# ── Availability rules / validation ────────────────────────────────


def test_validation_accepts_new_5c_features():
    feats = [
        "eeg_regime",
        "neg_price_frac_30d_d1",
        "neg_price_frac_90d_d1",
        "neg_price_depth_30d_d1",
        "pct_prog_solar",
        "pct_prog_wind_on",
        "pct_prog_wind_off",
    ]
    errors = validate_features(feats)
    assert errors == []


def test_validation_rejects_bare_neg_price():
    # neg_price_* maps to availability offset=-1, so a bare name must be flagged.
    errors = validate_features(["neg_price_frac_30d"])
    assert len(errors) == 1
    assert "Bare column name" in errors[0].reason
