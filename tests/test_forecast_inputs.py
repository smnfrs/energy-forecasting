"""Tests for source-neutral forecast input construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.features.forecast_inputs import (
    build_forecast_columns,
    forecast_source_counts,
    forecast_source_labels,
)

ARTIFACT_FILES = {
    "wind_onshore_DE_NATIONAL.parquet": "wind_on",
    "wind_offshore_DE_NATIONAL.parquet": "wind_off",
    "solar_DE_NATIONAL.parquet": "solar",
    "load_DE_NATIONAL.parquet": "load",
    "gen_load_diff_DE_NATIONAL.parquet": "diff",
}


def _write_artifacts(root, index, values):
    for filename, key in ARTIFACT_FILES.items():
        pd.DataFrame({"y_pred": values[key]}, index=index).to_parquet(root / filename)


def _raw_smard_frame(index):
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


def test_forecast_column_derivations_prefer_own_artifacts(tmp_path):
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    artifact_idx = idx.tz_localize("Europe/Berlin").tz_convert("UTC")
    _write_artifacts(
        tmp_path,
        artifact_idx,
        {
            "wind_on": [10.0, 11.0],
            "wind_off": [5.0, 5.0],
            "solar": [8.0, 8.0],
            "load": [50.0, 51.0],
            "diff": [3.0, 3.0],
        },
    )

    out = build_forecast_columns(_raw_smard_frame(idx), forecast_root=tmp_path)

    assert out["forecast_gen_wind_pv"].iloc[0] == pytest.approx(23.0)
    assert out["forecast_gen_total"].iloc[0] == pytest.approx(53.0)
    assert out["forecast_gen_other"].iloc[0] == pytest.approx(30.0)
    assert out["forecast_residual_load"].iloc[0] == pytest.approx(27.0)
    assert out.attrs["forecast_source_counts"]["own"] == 2


def test_waterfall_falls_back_to_smard_when_no_artifacts(tmp_path):
    idx = pd.date_range("2025-01-01", periods=2, freq="h")

    out = build_forecast_columns(_raw_smard_frame(idx), forecast_root=tmp_path)

    assert out["forecast_load"].iloc[0] == pytest.approx(500.0)
    assert out["forecast_gen_other"].iloc[0] == pytest.approx(300.0)
    assert out.attrs["forecast_source_counts"]["smard"] == 2


def test_waterfall_falls_back_to_actuals_when_smard_missing(tmp_path):
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    df = _raw_smard_frame(idx).drop(columns=[c for c in _raw_smard_frame(idx).columns if c.startswith("prognost")])

    out = build_forecast_columns(df, forecast_root=tmp_path)

    assert out["forecast_load"].iloc[0] == pytest.approx(50.0)
    assert out["forecast_gen_total"].iloc[0] == pytest.approx(53.0)
    assert out["forecast_gen_other"].iloc[0] == pytest.approx(30.0)
    assert out["forecast_residual_load"].iloc[0] == pytest.approx(27.0)
    assert out.attrs["forecast_source_counts"]["actual"] == 2


def test_partial_coverage_row_uses_next_coherent_layer(tmp_path):
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    artifact_idx = idx.tz_localize("Europe/Berlin").tz_convert("UTC")
    _write_artifacts(
        tmp_path,
        artifact_idx,
        {
            "wind_on": [10.0, 11.0],
            "wind_off": [5.0, 5.0],
            "solar": [8.0, 8.0],
            "load": [np.nan, 51.0],
            "diff": [3.0, 3.0],
        },
    )

    out = build_forecast_columns(_raw_smard_frame(idx), forecast_root=tmp_path)

    assert out["forecast_gen_wind_on"].iloc[0] == pytest.approx(100.0)
    assert out["forecast_load"].iloc[0] == pytest.approx(500.0)
    assert out["forecast_gen_wind_on"].iloc[1] == pytest.approx(11.0)
    assert out.attrs["forecast_source_counts"] == {"own": 1, "smard": 1, "actual": 0, "missing": 0}


def test_strict_mode_raises_on_missing_source(tmp_path):
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    artifact_idx = idx.tz_localize("Europe/Berlin").tz_convert("UTC")
    _write_artifacts(
        tmp_path,
        artifact_idx,
        {
            "wind_on": [10.0, 11.0],
            "wind_off": [5.0, 5.0],
            "solar": [8.0, 8.0],
            "load": [np.nan, 51.0],
            "diff": [3.0, 3.0],
        },
    )

    with pytest.raises(RuntimeError, match="strict coverage failed"):
        build_forecast_columns(_raw_smard_frame(idx), strict_index=idx, forecast_root=tmp_path)


def test_strict_mode_spring_forward_interpolates_missing_0200(tmp_path):
    strict_idx = pd.date_range("2025-03-30 00:00", "2025-03-30 23:00", freq="h")
    utc_idx = pd.date_range("2025-03-29 23:00", "2025-03-30 21:00", freq="h", tz="UTC")
    local_hours = utc_idx.tz_convert("Europe/Berlin").hour.astype(float)
    _write_artifacts(
        tmp_path,
        utc_idx,
        {
            "wind_on": local_hours,
            "wind_off": np.ones(len(utc_idx)),
            "solar": np.ones(len(utc_idx)),
            "load": local_hours,
            "diff": np.ones(len(utc_idx)),
        },
    )

    out = build_forecast_columns(_raw_smard_frame(strict_idx), strict_index=strict_idx, forecast_root=tmp_path)

    assert len(out.loc[strict_idx]) == 24
    assert out.loc[pd.Timestamp("2025-03-30 02:00"), "forecast_load"] == pytest.approx(2.0)
    assert out.loc[strict_idx, "forecast_load"].notna().all()


def test_forecast_source_labels_match_waterfall(tmp_path):
    idx = pd.date_range("2025-01-01", periods=3, freq="h")
    _write_artifacts(
        tmp_path,
        idx,
        {
            "wind_on": [10.0, np.nan, np.nan],
            "wind_off": [5.0, 5.0, np.nan],
            "solar": [8.0, 8.0, np.nan],
            "load": [50.0, 51.0, np.nan],
            "diff": [3.0, 3.0, np.nan],
        },
    )
    raw = _raw_smard_frame(idx)
    raw.loc[idx[2], [c for c in raw.columns if c.startswith("prognost")]] = np.nan

    labels = forecast_source_labels(raw, forecast_root=tmp_path)

    assert labels.tolist() == ["own", "smard", "actual"]
    assert forecast_source_counts(labels, idx) == {
        "own": 1,
        "smard": 1,
        "actual": 1,
        "missing": 0,
    }


def test_forecast_source_counts_treats_uncovered_reindex_as_missing(tmp_path):
    idx = pd.date_range("2025-01-01", periods=1, freq="h")
    labels = pd.Series(["own"], index=idx)
    requested = pd.date_range("2025-01-01", periods=2, freq="h")

    assert forecast_source_counts(labels, requested) == {
        "own": 1,
        "smard": 0,
        "actual": 0,
        "missing": 1,
    }
