"""Tests for the FastAPI endpoints using TestClient with mocked static data."""

import json

import pytest
from fastapi.testclient import TestClient


def _make_forecast_json(target, region, unit, n_hours, issued_at="2026-06-30T08:00:00Z"):
    import pandas as pd

    idx = pd.date_range("2026-07-01 00:00", periods=n_hours, freq="h")
    return {
        "target": target,
        "region": region,
        "issued_at": issued_at,
        "horizon_hours": n_hours,
        "unit": unit,
        "ensemble_method": "slsqp_optimized",
        "model_count": 5,
        "forecasts": [
            {
                "timestamp": ts.isoformat(),
                "forecast": 100.0 + i,
                "forecast_lower": 80.0 + i,
                "forecast_upper": 120.0 + i,
            }
            for i, ts in enumerate(idx)
        ],
    }


def _make_metadata_json():
    return {
        "target": "price",
        "ensemble_method": "slsqp_optimized",
        "models": [
            {"name": "LGBMRegressor__fs_shap_top90", "category": "lgbm", "weight": 0.346},
            {"name": "XGBRegressor__fs_rfecv_optimum", "category": "xgboost", "weight": 0.197},
        ],
        "holdout_mae": 11.148,
        "holdout_rmse": 18.2,
        "pi_coverage": 0.9,
        "last_retrain": "2026-06-30T08:00:00Z",
        "conformal_quantile": 24.4,
    }


@pytest.fixture
def deploy_data(tmp_path, monkeypatch):
    """Set up mock deploy/data/ directory and patch deps to use it."""
    import energy_forecasting.api.dependencies as deps

    price_json = _make_forecast_json("price", "DE_LU", "EUR/MWh", 24)
    gen_json = _make_forecast_json("wind_onshore", "DE_NATIONAL", "MW", 168)
    load_json = _make_forecast_json("load", "DE_NATIONAL", "MW", 168)
    meta_json = _make_metadata_json()
    history_json = {
        "target": "price",
        "forecasts": [price_json],
        "count": 1,
    }

    (tmp_path / "gen_load").mkdir()

    (tmp_path / "price_forecast.json").write_text(json.dumps(price_json))
    (tmp_path / "gen_load" / "wind_onshore_national.json").write_text(json.dumps(gen_json))
    (tmp_path / "gen_load" / "wind_offshore_national.json").write_text(
        json.dumps(_make_forecast_json("wind_offshore", "DE_NATIONAL", "MW", 168))
    )
    (tmp_path / "gen_load" / "solar_national.json").write_text(
        json.dumps(_make_forecast_json("solar", "DE_NATIONAL", "MW", 168))
    )
    (tmp_path / "gen_load" / "load_national.json").write_text(json.dumps(load_json))
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta_json))
    (tmp_path / "forecast_history.json").write_text(json.dumps(history_json))

    monkeypatch.setattr(deps, "PRICE_FORECAST_PATH", tmp_path / "price_forecast.json")
    monkeypatch.setattr(deps, "HISTORY_PATH", tmp_path / "forecast_history.json")
    monkeypatch.setattr(deps, "METADATA_PATH", tmp_path / "model_metadata.json")
    monkeypatch.setattr(deps, "GEN_LOAD_DATA_DIR", tmp_path / "gen_load")
    monkeypatch.setattr(deps, "ERRORS_DIR", tmp_path / "errors")
    monkeypatch.setattr(
        deps,
        "_GEN_LOAD_FILES",
        {
            "wind_onshore": tmp_path / "gen_load" / "wind_onshore_national.json",
            "wind_offshore": tmp_path / "gen_load" / "wind_offshore_national.json",
            "solar": tmp_path / "gen_load" / "solar_national.json",
            "load": tmp_path / "gen_load" / "load_national.json",
        },
    )
    monkeypatch.setattr(deps, "count_model_files", lambda: 5)
    monkeypatch.setattr(deps, "is_data_available", lambda: True)

    return tmp_path


@pytest.fixture
def client(deploy_data):
    from energy_forecasting.api.app import app

    return TestClient(app)


def test_health_healthy(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] == 5
    assert data["data_available"] is True


def test_price_forecast(client):
    resp = client.get("/forecast/price")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"] == "price"
    assert len(data["forecasts"]) == 24
    assert data["forecasts"][0]["forecast"] == pytest.approx(100.0)
    assert "forecast_lower" in data["forecasts"][0]


def test_generation_forecast_wind_onshore(client):
    resp = client.get("/forecast/generation/wind_onshore")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"] == "wind_onshore"
    assert len(data["forecasts"]) == 168


def test_generation_forecast_invalid_type(client):
    resp = client.get("/forecast/generation/nuclear")
    assert resp.status_code == 422


def test_load_forecast(client):
    resp = client.get("/forecast/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"] == "load"
    assert len(data["forecasts"]) == 168


def test_forecast_history(client):
    resp = client.get("/forecast/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"] == "price"
    assert data["count"] == 1
    assert len(data["forecasts"]) == 1


def test_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ensemble_method"] == "slsqp_optimized"
    assert len(data["models"]) == 2
    assert data["holdout_mae"] == pytest.approx(11.148)


def test_performance_no_errors(client):
    resp = client.get("/models/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"] == "price"
    assert data["blend_errors"] == []
