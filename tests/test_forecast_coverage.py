"""Tests for forecast artifact coverage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.features.forecast_coverage import (
    assert_monthly_artifact_coverage,
    assert_price_holdout_forecast_coverage,
)

ARTIFACT_FILES = {
    "wind_onshore_DE_NATIONAL.parquet": "wind_on",
    "wind_offshore_DE_NATIONAL.parquet": "wind_off",
    "solar_DE_NATIONAL.parquet": "solar",
    "load_DE_NATIONAL.parquet": "load",
    "gen_load_diff_DE_NATIONAL.parquet": "diff",
}


def _raw_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prognostizierter_verbrauch_gesamt": 500.0,
            "prognostizierte_erzeugung_onshore": 100.0,
            "prognostizierte_erzeugung_offshore": 50.0,
            "prognostizierte_erzeugung_photovoltaik": 80.0,
            "prognostizierte_erzeugung_wind_und_photovoltaik": 230.0,
            "prognostizierte_erzeugung_gesamt": 530.0,
            "prognostizierte_erzeugung_sonstige": 300.0,
            "prognostizierter_verbrauch_residuallast": 270.0,
            "stromverbrauch_gesamt_(netzlast)": 50.0,
            "stromerzeugung_wind_onshore": 10.0,
            "stromerzeugung_wind_offshore": 5.0,
            "stromerzeugung_photovoltaik": 8.0,
            "stromerzeugung_gesamt": 53.0,
        },
        index=index,
    )


def _write_artifacts(root, index: pd.DatetimeIndex, *, missing_load_mask=None) -> None:
    values = {
        "wind_on": np.full(len(index), 10.0),
        "wind_off": np.full(len(index), 5.0),
        "solar": np.full(len(index), 8.0),
        "load": np.full(len(index), 50.0),
        "diff": np.full(len(index), 3.0),
    }
    if missing_load_mask is not None:
        values["load"] = values["load"].copy()
        values["load"][missing_load_mask] = np.nan
    for filename, key in ARTIFACT_FILES.items():
        pd.DataFrame({"y_pred": values[key]}, index=index).to_parquet(root / filename)


def test_price_holdout_guard_uses_exact_dataset_holdout(tmp_path):
    idx = pd.date_range("2025-01-01", periods=60 * 24, freq="h")
    raw = _raw_frame(idx)
    _write_artifacts(tmp_path, idx)

    report = assert_price_holdout_forecast_coverage(
        idx,
        raw,
        holdout_days=7,
        min_own_fraction=0.95,
        forecast_root=tmp_path,
        context="unit",
    )

    assert report["own_fraction"] == pytest.approx(1.0)
    assert report["counts"]["own"] == 7 * 24


def test_price_holdout_guard_rejects_smard_fallback_in_holdout(tmp_path):
    idx = pd.date_range("2025-01-01", periods=60 * 24, freq="h")
    raw = _raw_frame(idx)
    missing = idx >= idx[-48]
    _write_artifacts(tmp_path, idx, missing_load_mask=missing)

    with pytest.raises(RuntimeError, match="own_fraction=.*< 95.00%"):
        assert_price_holdout_forecast_coverage(
            idx,
            raw,
            holdout_days=7,
            min_own_fraction=0.95,
            forecast_root=tmp_path,
            context="unit",
        )


def test_monthly_artifact_coverage_rejects_hole(tmp_path):
    idx = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="h")
    raw = _raw_frame(idx)
    missing = idx.month == 2
    _write_artifacts(tmp_path, idx, missing_load_mask=missing)
    merged_path = tmp_path / "merged.parquet"
    raw.to_parquet(merged_path)

    with pytest.raises(RuntimeError, match="2025-02"):
        assert_monthly_artifact_coverage(
            merged_path=merged_path,
            start="2025-01-01",
            min_own_fraction=0.95,
            forecast_root=tmp_path,
        )
