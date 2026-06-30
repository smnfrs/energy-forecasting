"""Tests for deploy/publish.py — JSON schema consistency with Pydantic models."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.api.schemas import ForecastResponse, HourlyForecast


def _price_df(n=24) -> pd.DataFrame:
    idx = pd.date_range("2026-06-30 00:00", periods=n, freq="h")
    return pd.DataFrame(
        {
            "y_pred": np.linspace(80, 150, n),
            "y_lower": np.linspace(60, 130, n),
            "y_upper": np.linspace(100, 170, n),
        },
        index=idx,
    )


def _gen_df(n=168) -> pd.DataFrame:
    idx = pd.date_range("2026-06-30 00:00", periods=n, freq="h")
    return pd.DataFrame(
        {
            "y_pred": np.ones(n) * 5000.0,
            "y_lower": np.ones(n) * 4500.0,
            "y_upper": np.ones(n) * 5500.0,
        },
        index=idx,
    )


def test_price_json_is_valid_forecast_response(tmp_path, monkeypatch):
    """Price JSON must deserialize to a valid ForecastResponse."""
    import energy_forecasting.deploy.publish as pub

    # Patch DEPLOY_DATA_DIR to tmp_path
    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", tmp_path)
    monkeypatch.setattr(pub, "PRICE_FORECAST_PATH", tmp_path / "price_forecast.json")
    monkeypatch.setattr(pub, "HISTORY_PATH", tmp_path / "forecast_history.json")
    monkeypatch.setattr(pub, "GEN_LOAD_DATA_DIR", tmp_path / "gen_load")

    # Patch model_store to avoid reading real config
    import energy_forecasting.deploy.model_store as ms

    fake_config = {
        "ensemble": {"method": "slsqp_optimized", "weights": {}},
        "models": [],
        "conformal_quantile": 24.4,
        "pi_coverage": 0.9,
        "metrics": {"mae": 11.1, "rmse": 18.2},
    }
    monkeypatch.setattr(ms, "load_ensemble_config", lambda: fake_config)
    monkeypatch.setattr(ms, "production_model_names", lambda c: [])

    price_df = _price_df()
    pub.write_price_forecast(price_df, issued_at="2026-06-30T08:00:00Z")

    raw = json.loads((tmp_path / "price_forecast.json").read_text())
    resp = ForecastResponse(**raw)
    assert resp.target == "price"
    assert len(resp.forecasts) == 24
    assert all(isinstance(f, HourlyForecast) for f in resp.forecasts)
    assert resp.forecasts[0].forecast_lower is not None


def test_gen_load_json_valid(tmp_path, monkeypatch):
    """Gen/load JSON must deserialize to a valid ForecastResponse."""
    import energy_forecasting.deploy.publish as pub

    monkeypatch.setattr(pub, "GEN_LOAD_DATA_DIR", tmp_path / "gen_load")

    gen_load_results = {
        ("wind_onshore", "DE_NATIONAL"): _gen_df(),
        ("wind_offshore", "DE_NATIONAL"): _gen_df(),
        ("solar", "DE_NATIONAL"): _gen_df(),
        ("load", "DE_NATIONAL"): _gen_df(),
    }
    pub.write_gen_load_forecasts(gen_load_results, issued_at="2026-06-30T08:00:00Z")

    path = tmp_path / "gen_load" / "wind_onshore_national.json"
    assert path.exists()
    raw = json.loads(path.read_text())
    resp = ForecastResponse(**raw)
    assert resp.target == "wind_onshore"
    assert len(resp.forecasts) == 168


def test_history_appends_and_trims(tmp_path, monkeypatch):
    """History file should deduplicate and retain last 30 entries."""
    import energy_forecasting.deploy.publish as pub

    monkeypatch.setattr(pub, "HISTORY_PATH", tmp_path / "forecast_history.json")
    monkeypatch.setattr(pub, "HISTORY_DAYS", 3)

    price_df = _price_df()
    pub._append_price_history(price_df, "2026-06-28T08:00:00Z")
    pub._append_price_history(price_df, "2026-06-29T08:00:00Z")
    pub._append_price_history(price_df, "2026-06-30T08:00:00Z")
    # Add one more to trigger trim
    pub._append_price_history(price_df, "2026-07-01T08:00:00Z")

    history = json.loads((tmp_path / "forecast_history.json").read_text())
    assert history["count"] == 3  # trimmed to HISTORY_DAYS
    dates = [f["issued_at"][:10] for f in history["forecasts"]]
    assert "2026-06-28" not in dates  # oldest dropped
