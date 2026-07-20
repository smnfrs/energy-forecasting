"""Tests for deploy/retrain.py ensemble mode routing."""

from __future__ import annotations

from pathlib import Path


def _config() -> dict:
    return {
        "ensemble": {"weights": {"m0": 0.7, "m1": 0.3, "m2": 0.0}},
        "metrics": {"mae": 10.0},
        "models": [
            {
                "name": "m0",
                "run_id": "old0",
                "model_type": "LGBMRegressor",
                "feature_version": "fv",
                "config": {},
            },
            {
                "name": "m1",
                "run_id": "old1",
                "model_type": "XGBRegressor",
                "feature_version": "fv",
                "config": {},
            },
            {
                "name": "m2",
                "run_id": "old2",
                "model_type": "CatBoostRegressor",
                "feature_version": "fv",
                "config": {},
            },
        ],
    }


def test_build_retrain_ensemble_reweight_uses_fixed_production_members(monkeypatch):
    import energy_forecasting.deploy.retrain as retrain

    calls = {}

    def fake_build(model_runs, **kwargs):
        calls["names"] = [mr.name for mr in model_runs]
        calls["candidate_model_names"] = kwargs["candidate_model_names"]
        calls["strict_categories"] = kwargs["strict_categories"]

        class Result:
            selected_models = ["m0", "m1"]
            selected_model_runs = model_runs
            alignment_metadata = {"oof_row_count": 100, "holdout_row_count": 24}
            candidate_metrics = _FakeFrame([])
            comparison = _FakeFrame([])
            metrics = {"mae": 9.5}
            ensemble = _FakeEnsemble()

        return Result()

    monkeypatch.setattr("energy_forecasting.modeling.ensemble.build_production_ensemble", fake_build)
    monkeypatch.setattr("energy_forecasting.modeling.ensemble.ensemble_config_dict", fake_config_dict)
    cfg, mae = retrain._build_retrain_ensemble(
        _config(),
        {"m0": "new0", "m1": "new1"},
        mode="reweight",
    )
    assert mae == 9.5
    assert cfg["metrics"]["mae"] == 9.5
    assert calls["names"] == ["m0", "m1"]
    assert calls["candidate_model_names"] == {"m0", "m1"}
    assert calls["strict_categories"] is False
    assert {entry["name"] for entry in cfg["models"]} == {"m0", "m1", "m2"}
    assert {entry["run_id"] for entry in cfg["models"]} == {"new0", "new1", "old2"}
    assert cfg["artifact_generation"] == {
        "mode": "steady_state",
        "fresh_run_names": ["m0", "m1"],
        "all_candidates_fresh": False,
    }


def test_build_retrain_ensemble_reselection_uses_all_config_models(monkeypatch):
    import energy_forecasting.deploy.retrain as retrain

    calls = {}

    def fake_build(model_runs, **kwargs):
        calls["names"] = [mr.name for mr in model_runs]
        calls["candidate_model_names"] = kwargs["candidate_model_names"]
        calls["strict_categories"] = kwargs["strict_categories"]

        class Result:
            selected_models = ["m0", "m2"]
            selected_model_runs = model_runs
            alignment_metadata = {"oof_row_count": 100, "holdout_row_count": 24}
            candidate_metrics = _FakeFrame([])
            comparison = _FakeFrame([])
            metrics = {"mae": 8.5}
            ensemble = _FakeEnsemble()

        return Result()

    monkeypatch.setattr("energy_forecasting.modeling.ensemble.build_production_ensemble", fake_build)
    monkeypatch.setattr("energy_forecasting.modeling.ensemble.ensemble_config_dict", fake_config_dict)
    cfg, mae = retrain._build_retrain_ensemble(
        _config(),
        {"m0": "new0", "m1": "new1", "m2": "new2"},
        mode="reselection",
    )
    assert mae == 8.5
    assert cfg["metrics"]["mae"] == 8.5
    assert calls["names"] == ["m0", "m1", "m2"]
    assert calls["candidate_model_names"] is None
    assert calls["strict_categories"] is True
    assert {entry["run_id"] for entry in cfg["models"]} == {"new0", "new1", "new2"}
    assert cfg["artifact_generation"] == {
        "mode": "bootstrap_reselection",
        "fresh_run_names": ["m0", "m1", "m2"],
        "all_candidates_fresh": True,
    }


class _FakeFrame:
    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return self._rows

    def drop(self, columns):
        return self


class _FakeEnsemble:
    method = "inverse_mae"
    model_names = ["m0", "m1"]


def fake_config_dict(ensemble, *, base_runs, metrics):
    return {
        "ensemble": {"method": ensemble.method, "weights": {"m0": 0.5, "m1": 0.5}},
        "metrics": metrics,
        "models": [{"name": name, **info} for name, info in base_runs.items()],
    }


def test_rebuild_price_dataset_from_merged_uses_fresh_rows(tmp_path, monkeypatch):
    import energy_forecasting.deploy.retrain as retrain
    import numpy as np
    import pandas as pd
    from energy_forecasting.modeling import datasets

    idx = pd.date_range("2026-07-01", periods=5, freq="h")
    merged = pd.DataFrame({"target_price": [1.0, 2.0, np.nan, 4.0, 5.0]}, index=idx)

    orig_read_parquet = pd.read_parquet
    monkeypatch.setattr(retrain, "_price_feature_columns", lambda feature_version: ["x"])
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: merged if Path(path).name == "merged.parquet" else orig_read_parquet(path),
    )
    monkeypatch.setattr(
        "energy_forecasting.features.forecast_inputs.build_forecast_columns",
        lambda df: df,
    )
    monkeypatch.setattr(
        "energy_forecasting.features.engine.engineer_features",
        lambda df, feature_list, validate=False: pd.DataFrame(
            {"x": [10.0, 11.0, 12.0, np.nan, 14.0]},
            index=df.index,
        ),
    )
    monkeypatch.setattr(datasets, "DATASET_DIR", tmp_path)

    out = retrain.rebuild_price_dataset_from_merged("fv", merged_path=tmp_path / "merged.parquet")

    saved = pd.read_parquet(out)
    assert out == tmp_path / "price_fv.parquet"
    assert list(saved.columns) == ["x", "target_price__target"]
    assert list(saved.index) == [idx[0], idx[1], idx[4]]
