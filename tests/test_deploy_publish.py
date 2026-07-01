"""Tests for deploy/publish.py — JSON schema consistency and new Stage 7 functions."""

import json

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


# ── Stage 7 additions ─────────────────────────────────────────────────────────

def _merged_parquet(path, n_days=32, partial_day=True):
    """Write a merged.parquet with n_days complete days + optional partial tail."""
    rows = []
    base = pd.Timestamp("2026-05-30 00:00")
    for d in range(n_days):
        for h in range(24):
            rows.append({"ts": base + pd.Timedelta(days=d, hours=h), "target_price": float(d * 100 + h)})
    if partial_day:
        rows.append({"ts": base + pd.Timedelta(days=n_days, hours=0), "target_price": 999.0})
    df = pd.DataFrame(rows).set_index("ts")
    df.to_parquet(path)
    return df


def test_write_actuals_filters_then_trims(tmp_path, monkeypatch):
    """Complete days are filtered before trimming to 30; partial tail excluded."""
    import energy_forecasting.deploy.publish as pub
    import energy_forecasting.config as cfg_mod

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    _merged_parquet(data_dir / "merged.parquet", n_days=32, partial_day=True)

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", tmp_path / "deploy")
    monkeypatch.setattr(cfg_mod, "PROCESSED_DATA_DIR", data_dir)

    pub.write_actuals()

    data = json.loads((tmp_path / "deploy" / "actuals.json").read_text())
    assert "days" in data and "count" in data
    assert data["count"] == 30
    for day in data["days"]:
        assert len(day["prices"]) == 24


def test_write_actuals_missing_parquet(tmp_path, monkeypatch):
    """write_actuals returns silently if merged.parquet is absent."""
    import energy_forecasting.deploy.publish as pub
    import energy_forecasting.config as cfg_mod

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", tmp_path / "deploy")
    monkeypatch.setattr(cfg_mod, "PROCESSED_DATA_DIR", tmp_path / "processed")

    pub.write_actuals()  # must not raise

    assert not (tmp_path / "deploy" / "actuals.json").exists()


def test_write_errors_summary(tmp_path, monkeypatch):
    """Aggregates errors/*.json sorted by date, takes last 30."""
    import energy_forecasting.deploy.publish as pub

    deploy_dir = tmp_path / "deploy"
    errors_dir = deploy_dir / "errors"
    errors_dir.mkdir(parents=True)

    for date, mae, rmse in [("2026-06-28", 10.0, 16.0), ("2026-06-29", 11.0, 17.0)]:
        (errors_dir / f"{date}.json").write_text(
            json.dumps({"date": date, "mae": mae, "rmse": rmse})
        )

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", deploy_dir)
    monkeypatch.setattr(pub, "ERRORS_DIR", errors_dir)

    pub.write_errors_summary()

    summary = json.loads((deploy_dir / "errors_summary.json").read_text())
    assert summary["dates"] == ["2026-06-28", "2026-06-29"]
    assert summary["mae"] == [10.0, 11.0]
    assert summary["rmse"] == [16.0, 17.0]


def test_write_errors_summary_trims_to_30(tmp_path, monkeypatch):
    """Only last 30 date-named error files are included."""
    import energy_forecasting.deploy.publish as pub

    deploy_dir = tmp_path / "deploy"
    errors_dir = deploy_dir / "errors"
    errors_dir.mkdir(parents=True)

    base = pd.Timestamp("2026-05-01")
    for i in range(35):
        date = (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        (errors_dir / f"{date}.json").write_text(
            json.dumps({"date": date, "mae": float(i), "rmse": float(i * 1.5)})
        )

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", deploy_dir)
    monkeypatch.setattr(pub, "ERRORS_DIR", errors_dir)

    pub.write_errors_summary()

    summary = json.loads((deploy_dir / "errors_summary.json").read_text())
    assert len(summary["dates"]) == 30
    assert summary["dates"][0] == "2026-05-06"  # oldest 5 dropped


def _tso_parquet(path, tso_suffix, n_complete_days=8):
    """Write a minimal processed TSO parquet with wind_onshore, solar, load columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2026-06-23 00:00", tz="UTC")
    rows = []
    for d in range(n_complete_days):
        for h in range(24):
            rows.append({
                "time": base + pd.Timedelta(days=d, hours=h),
                f"wind_onshore{tso_suffix}": float(d * 100 + h),
                f"wind_offshore{tso_suffix}": float(d * 50 + h) if tso_suffix in ("_50hz", "_tenn") else None,
                f"solar{tso_suffix}": float(d * 80 + h),
                f"load{tso_suffix}": float(d * 200 + h),
            })
    df = pd.DataFrame(rows).set_index("time")
    df.to_parquet(path)
    return df


def test_write_gen_load_forecasts_writes_tso_files(tmp_path, monkeypatch):
    """Per-TSO JSON files are written alongside national files."""
    import energy_forecasting.deploy.publish as pub

    monkeypatch.setattr(pub, "GEN_LOAD_DATA_DIR", tmp_path / "gen_load")

    gen_load_results = {
        ("wind_onshore", "DE_NATIONAL"): _gen_df(),
        ("wind_onshore", "DE_50HZ"): _gen_df(),
        ("wind_onshore", "DE_AMPRION"): _gen_df(),
        ("load", "DE_NATIONAL"): _gen_df(),
        ("load", "DE_CREOS"): _gen_df(),
    }
    pub.write_gen_load_forecasts(gen_load_results, issued_at="2026-06-30T08:00:00Z")

    gl_dir = tmp_path / "gen_load"
    assert (gl_dir / "wind_onshore_national.json").exists()
    assert (gl_dir / "wind_onshore_50hz.json").exists()
    assert (gl_dir / "wind_onshore_amprion.json").exists()
    assert (gl_dir / "load_national.json").exists()
    assert (gl_dir / "load_creos.json").exists()
    # gen_load_diff should not be written
    assert not list(gl_dir.glob("gen_load_diff*.json"))


def test_write_gen_load_actuals_writes_file(tmp_path, monkeypatch):
    """write_gen_load_actuals produces gen_load_actuals.json from TSO parquets."""
    import energy_forecasting.deploy.publish as pub
    import energy_forecasting.config as cfg_mod

    tso_dir = tmp_path / "processed" / "tso"
    _tso_parquet(tso_dir / "50Hertz.parquet", "_50hz", n_complete_days=8)
    _tso_parquet(tso_dir / "Amprion.parquet", "_ampr", n_complete_days=8)
    _tso_parquet(tso_dir / "TenneT.parquet", "_tenn", n_complete_days=8)
    _tso_parquet(tso_dir / "TransnetBW.parquet", "_tran", n_complete_days=8)
    _tso_parquet(tso_dir / "Creos.parquet", "_lu", n_complete_days=8)

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", tmp_path / "deploy")
    monkeypatch.setattr(cfg_mod, "PROCESSED_DATA_DIR", tmp_path / "processed")

    pub.write_gen_load_actuals()

    out = tmp_path / "deploy" / "gen_load_actuals.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "wind_onshore" in data
    assert "load" in data
    days = data["wind_onshore"]["days"]
    assert len(days) <= 7
    assert all(len(d["values"]) == 24 for d in days)


def test_write_gen_load_actuals_missing_tso_dir(tmp_path, monkeypatch):
    """write_gen_load_actuals silently skips if tso dir is absent."""
    import energy_forecasting.deploy.publish as pub
    import energy_forecasting.config as cfg_mod

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", tmp_path / "deploy")
    monkeypatch.setattr(cfg_mod, "PROCESSED_DATA_DIR", tmp_path / "processed")

    pub.write_gen_load_actuals()  # must not raise
    assert not (tmp_path / "deploy" / "gen_load_actuals.json").exists()


def test_write_outputs_always_calls_errors_summary(tmp_path, monkeypatch):
    """write_outputs rebuilds errors_summary.json unconditionally."""
    import energy_forecasting.deploy.publish as pub
    import energy_forecasting.deploy.model_store as ms

    deploy_dir = tmp_path / "deploy"
    errors_dir = deploy_dir / "errors"
    errors_dir.mkdir(parents=True)
    (errors_dir / "2026-06-29.json").write_text(
        json.dumps({"date": "2026-06-29", "mae": 9.9, "rmse": 15.5})
    )

    monkeypatch.setattr(pub, "DEPLOY_DATA_DIR", deploy_dir)
    monkeypatch.setattr(pub, "PRICE_FORECAST_PATH", deploy_dir / "price_forecast.json")
    monkeypatch.setattr(pub, "HISTORY_PATH", deploy_dir / "forecast_history.json")
    monkeypatch.setattr(pub, "METADATA_PATH", deploy_dir / "model_metadata.json")
    monkeypatch.setattr(pub, "GEN_LOAD_DATA_DIR", deploy_dir / "gen_load")
    monkeypatch.setattr(pub, "ERRORS_DIR", errors_dir)

    import energy_forecasting.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROCESSED_DATA_DIR", tmp_path / "processed")

    fake_config = {
        "ensemble": {"method": "slsqp_optimized", "weights": {}},
        "models": [],
        "conformal_quantile": 24.4,
        "pi_coverage": 0.9,
        "metrics": {"mae": 11.1, "rmse": 18.2},
    }
    monkeypatch.setattr(ms, "load_ensemble_config", lambda: fake_config)
    monkeypatch.setattr(ms, "production_model_names", lambda c: [])

    pub.write_outputs(_price_df(), {}, issued_at="2026-06-30T08:00:00Z")

    assert (deploy_dir / "errors_summary.json").exists()
    summary = json.loads((deploy_dir / "errors_summary.json").read_text())
    assert summary["dates"] == ["2026-06-29"]
