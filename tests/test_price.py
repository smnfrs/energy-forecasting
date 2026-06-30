"""Tests for modeling/price.py — orchestrator helpers.

End-to-end ``run_price_pipeline`` requires MLflow + Optuna + actual model
training, so it is exercised by ``train price`` runs rather than here.
These tests cover the smaller helpers: dataset prep dispatch, prediction
stacking, the feature-list registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.modeling.price import (
    FEATURE_LISTS,
    PRICE_TARGET,
    _ModelRun,
    _stack_predictions,
)


def test_feature_list_registry_contains_known_versions():
    assert set(FEATURE_LISTS) >= {"slim", "full", "max"}
    for fv, feats in FEATURE_LISTS.items():
        assert isinstance(feats, list)
        assert len(feats) > 0
        assert all(isinstance(f, str) for f in feats)


def test_price_target_constant():
    # PRICE_TARGET is what the orchestrator reads off the merged dataset.
    # If this changes, prepare_dataset gets the wrong column.
    assert PRICE_TARGET == "target_price"


# ── _stack_predictions ───────────────────────────────────────────


def _fake_artifact_pair(
    tmp_path: Path,
    name: str,
    oof_idx: pd.DatetimeIndex,
    hold_idx: pd.DatetimeIndex,
    oof_preds: np.ndarray,
    hold_preds: np.ndarray,
    y_oof: np.ndarray,
    y_hold: np.ndarray,
) -> Path:
    """Create the same artifact layout train_model writes."""
    run_dir = tmp_path / name / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"y_true": y_oof, "y_pred": oof_preds}, index=oof_idx,
    ).to_parquet(run_dir / "oof_predictions.parquet")
    pd.DataFrame(
        {
            "y_true": y_hold,
            "y_pred": hold_preds,
            "y_lower": np.nan,
            "y_upper": np.nan,
        },
        index=hold_idx,
    ).to_parquet(run_dir / "holdout_predictions.parquet")
    return tmp_path / name


def test_stack_predictions_returns_aligned_matrices(tmp_path, monkeypatch):
    """Mock _fetch_predictions to read from the temp artifact tree."""
    import energy_forecasting.modeling.price as price_module

    oof_idx = pd.date_range("2024-01-01", periods=200, freq="h")
    hold_idx = pd.date_range("2024-02-01", periods=80, freq="h")
    y_oof = np.arange(200, dtype=float)
    y_hold = np.arange(80, dtype=float)

    artifact_dirs: dict[str, Path] = {}
    for i, name in enumerate(["m0", "m1"]):
        oof_preds = y_oof + (i + 1)
        hold_preds = y_hold + (i + 1)
        artifact_dirs[name] = _fake_artifact_pair(
            tmp_path, name, oof_idx, hold_idx, oof_preds, hold_preds, y_oof, y_hold,
        )

    def fake_fetch(run_id):
        path = artifact_dirs[run_id] / "predictions"
        oof = pd.read_parquet(path / "oof_predictions.parquet")
        hold = pd.read_parquet(path / "holdout_predictions.parquet")
        return oof, hold

    monkeypatch.setattr(price_module, "_fetch_predictions", fake_fetch)

    model_runs = [
        _ModelRun(name="m0", run_id="m0", model_type="LGBMRegressor",
                  feature_version="slim", config={}),
        _ModelRun(name="m1", run_id="m1", model_type="Ridge",
                  feature_version="slim", config={}),
    ]
    preds_oof, y_oof_out, preds_hold, y_hold_out = _stack_predictions(model_runs)
    assert list(preds_oof.columns) == ["m0", "m1"]
    assert list(preds_hold.columns) == ["m0", "m1"]
    assert len(preds_oof) == 200
    assert len(preds_hold) == 80
    # Predictions came from the y + offset construction.
    np.testing.assert_allclose(preds_oof["m0"].to_numpy(), y_oof + 1)
    np.testing.assert_allclose(preds_oof["m1"].to_numpy(), y_oof + 2)
    np.testing.assert_array_equal(y_oof_out.to_numpy(), y_oof)
    np.testing.assert_array_equal(y_hold_out.to_numpy(), y_hold)


def test_stack_predictions_intersects_indices(tmp_path, monkeypatch):
    """When base models have different OOF coverage, the result should
    drop rows that aren't present in every model's predictions."""
    import energy_forecasting.modeling.price as price_module

    full_idx = pd.date_range("2024-01-01", periods=100, freq="h")
    short_idx = full_idx[20:]  # 80 rows — m1 has less OOF coverage
    hold_idx = pd.date_range("2024-02-01", periods=24, freq="h")

    y_full = np.arange(100, dtype=float)
    y_short = y_full[20:]
    y_hold = np.arange(24, dtype=float)

    artifact_dirs = {
        "m0": _fake_artifact_pair(
            tmp_path, "m0", full_idx, hold_idx,
            y_full + 1, y_hold + 1, y_full, y_hold,
        ),
        "m1": _fake_artifact_pair(
            tmp_path, "m1", short_idx, hold_idx,
            y_short + 2, y_hold + 2, y_short, y_hold,
        ),
    }

    def fake_fetch(run_id):
        path = artifact_dirs[run_id] / "predictions"
        oof = pd.read_parquet(path / "oof_predictions.parquet")
        hold = pd.read_parquet(path / "holdout_predictions.parquet")
        return oof, hold

    monkeypatch.setattr(price_module, "_fetch_predictions", fake_fetch)

    model_runs = [
        _ModelRun(name="m0", run_id="m0", model_type="LGBMRegressor",
                  feature_version="slim", config={}),
        _ModelRun(name="m1", run_id="m1", model_type="Ridge",
                  feature_version="slim", config={}),
    ]
    preds_oof, y_oof_out, preds_hold, y_hold_out = _stack_predictions(model_runs)
    assert len(preds_oof) == 80  # intersection size
    assert len(preds_hold) == 24
    # First row of the result aligns with m1's start (the shorter index).
    assert preds_oof.index[0] == short_idx[0]
