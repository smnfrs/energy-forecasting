"""Tests for deploy/model_store.py."""

import json

import pandas as pd
from energy_forecasting.deploy import model_store as ms


def test_export_price_feature_columns_writes_production_feature_versions(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    pd.DataFrame(
        {
            "forecast_load": [1.0],
            "price_lag_h24": [2.0],
            "target_price__target": [3.0],
            "__index_level_0__": [0],
        }
    ).to_parquet(dataset_dir / "price_fs_keep.parquet")
    pd.DataFrame({"prog_residual": [1.0], "target_price__target": [2.0]}).to_parquet(
        dataset_dir / "price_fs_zero_weight.parquet"
    )

    import energy_forecasting.modeling.datasets as datasets

    monkeypatch.setattr(datasets, "DATASET_DIR", dataset_dir)
    monkeypatch.setattr(ms, "PRICE_FEATURE_COLS_PATH", tmp_path / "price_feature_cols.json")

    config = {
        "ensemble": {
            "weights": {
                "model_keep": 1.0,
                "model_zero": 0.0,
            }
        },
        "models": [
            {"name": "model_keep", "feature_version": "fs_keep"},
            {"name": "model_zero", "feature_version": "fs_zero_weight"},
        ],
    }

    out = ms.export_price_feature_columns(config)

    assert out == tmp_path / "price_feature_cols.json"
    assert json.loads(out.read_text()) == {"fs_keep": ["forecast_load", "price_lag_h24"]}

def test_export_price_models_prunes_stale_joblibs(tmp_path, monkeypatch):
    price_dir = tmp_path / "price"
    price_dir.mkdir()
    stale = price_dir / "stale.joblib"
    stale.write_bytes(b"old")

    feature_cols_path = tmp_path / "price_feature_cols.json"

    def _stub_export_feature_cols(config):
        # Mirror production: write the manifest for the production feature
        # version(s) so the lockstep validation has a consistent artifact.
        feature_cols_path.write_text(json.dumps({"fs_keep": ["forecast_load"]}))
        return feature_cols_path

    monkeypatch.setattr(ms, "PRICE_MODELS_DIR", price_dir)
    monkeypatch.setattr(ms, "PRICE_FEATURE_COLS_PATH", feature_cols_path)
    monkeypatch.setattr(ms, "_download_model", lambda run_id: {"run_id": run_id})
    monkeypatch.setattr(ms, "_download_scaler", lambda run_id: None)
    monkeypatch.setattr(ms, "export_price_feature_columns", _stub_export_feature_cols)

    config = {
        "ensemble": {"weights": {"model_keep": 1.0, "model_zero": 0.0}},
        "models": [
            {"name": "model_keep", "run_id": "abc", "feature_version": "fs_keep"},
            {"name": "model_zero", "run_id": "def", "feature_version": "fs_zero"},
        ],
    }

    written = ms.export_price_models(config)

    assert not stale.exists()
    assert price_dir.joinpath("abc.joblib").exists()
    assert not price_dir.joinpath("def.joblib").exists()
    assert tmp_path / "price_feature_cols.json" in written

