"""Write forecast outputs to deploy/data/ for the FastAPI server.

Output structure:
    deploy/data/
    ├── price_forecast.json          ← ForecastResponse for today's price forecast
    ├── gen_load/
    │   ├── wind_onshore_national.json
    │   ├── wind_offshore_national.json
    │   ├── solar_national.json
    │   └── load_national.json
    ├── forecast_history.json        ← rolling 30-day price history
    ├── model_metadata.json          ← ensemble composition
    └── errors/
        └── {date}.json              ← daily MAE/RMSE once actuals are available

All files are loaded by the FastAPI server's dependency layer and served as-is.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config import DEPLOY_DATA_DIR, MODELS_DIR

GEN_LOAD_DATA_DIR = DEPLOY_DATA_DIR / "gen_load"
ERRORS_DIR = DEPLOY_DATA_DIR / "errors"
HISTORY_PATH = DEPLOY_DATA_DIR / "forecast_history.json"
METADATA_PATH = DEPLOY_DATA_DIR / "model_metadata.json"
PRICE_FORECAST_PATH = DEPLOY_DATA_DIR / "price_forecast.json"
HISTORY_DAYS = 30


def _ts(dt) -> str:
    """Format a timestamp as ISO 8601 string."""
    if isinstance(dt, pd.Timestamp):
        return dt.isoformat()
    return str(dt)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hourly_entries(df: pd.DataFrame) -> list[dict]:
    """Convert a forecast DataFrame to list of hourly dicts."""
    entries = []
    for ts, row in df.iterrows():
        entry: dict = {
            "timestamp": _ts(ts),
            "forecast": round(float(row["y_pred"]), 3),
        }
        if "y_lower" in row and not np.isnan(row["y_lower"]):
            entry["forecast_lower"] = round(float(row["y_lower"]), 3)
        if "y_upper" in row and not np.isnan(row["y_upper"]):
            entry["forecast_upper"] = round(float(row["y_upper"]), 3)
        entries.append(entry)
    return entries


def write_price_forecast(
    price_df: pd.DataFrame, issued_at: str | None = None
) -> None:
    """Write price_forecast.json."""
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    issued_at = issued_at or _now_utc()

    from energy_forecasting.deploy.model_store import (
        load_ensemble_config,
        production_model_names,
    )

    cfg = load_ensemble_config()
    prod_names = production_model_names(cfg)

    payload = {
        "target": "price",
        "region": "DE_LU",
        "issued_at": issued_at,
        "horizon_hours": len(price_df),
        "unit": "EUR/MWh",
        "ensemble_method": cfg["ensemble"]["method"],
        "model_count": len(prod_names),
        "forecasts": _hourly_entries(price_df),
    }
    PRICE_FORECAST_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(f"Written {PRICE_FORECAST_PATH}")


def _append_price_history(price_df: pd.DataFrame, issued_at: str) -> None:
    """Append today's price forecast to forecast_history.json (rolling 30d)."""
    entry = {
        "target": "price",
        "region": "DE_LU",
        "issued_at": issued_at,
        "horizon_hours": len(price_df),
        "unit": "EUR/MWh",
        "forecasts": _hourly_entries(price_df),
    }

    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text())
    else:
        history = {"target": "price", "forecasts": [], "count": 0}

    # Deduplicate by issued_at date
    forecasts = [
        f for f in history.get("forecasts", [])
        if f.get("issued_at", "")[:10] != issued_at[:10]
    ]
    forecasts.append(entry)
    # Keep last HISTORY_DAYS entries
    forecasts = sorted(forecasts, key=lambda f: f.get("issued_at", ""))
    forecasts = forecasts[-HISTORY_DAYS:]

    history["forecasts"] = forecasts
    history["count"] = len(forecasts)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    logger.info(f"Updated {HISTORY_PATH} ({len(forecasts)} entries)")


def write_gen_load_forecasts(
    gen_load_results: dict,
    issued_at: str | None = None,
) -> None:
    """Write national gen/load forecast JSONs."""
    GEN_LOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    issued_at = issued_at or _now_utc()

    target_to_file = {
        "wind_onshore": "wind_onshore_national.json",
        "wind_offshore": "wind_offshore_national.json",
        "solar": "solar_national.json",
        "load": "load_national.json",
    }
    units = {
        "wind_onshore": "MW",
        "wind_offshore": "MW",
        "solar": "MW",
        "load": "MW",
    }

    for target, filename in target_to_file.items():
        key = (target, "DE_NATIONAL")
        if key not in gen_load_results:
            logger.warning(f"No national forecast for {target}, skipping")
            continue
        df = gen_load_results[key]
        payload = {
            "target": target,
            "region": "DE_NATIONAL",
            "issued_at": issued_at,
            "horizon_hours": len(df),
            "unit": units[target],
            "forecasts": _hourly_entries(df),
        }
        out = GEN_LOAD_DATA_DIR / filename
        out.write_text(json.dumps(payload, indent=2))
        logger.info(f"Written {out}")


def write_model_metadata(issued_at: str | None = None) -> None:
    """Write model_metadata.json with ensemble composition."""
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    issued_at = issued_at or _now_utc()

    from energy_forecasting.deploy.model_store import (
        load_ensemble_config,
        production_model_names,
    )

    cfg = load_ensemble_config()
    prod_names = set(production_model_names(cfg))
    weights = cfg["ensemble"]["weights"]

    models = []
    for entry in cfg["models"]:
        if entry["name"] not in prod_names:
            continue
        # Derive category from model_type
        mt = entry.get("model_type", entry["name"])
        if "LGBM" in mt:
            cat = "lgbm"
        elif "XGB" in mt:
            cat = "xgboost"
        elif "CatBoost" in mt:
            cat = "catboost"
        else:
            cat = "linear"
        models.append({
            "name": entry["name"],
            "category": cat,
            "weight": round(weights.get(entry["name"], 0.0), 6),
            "run_id": entry["run_id"],
        })

    payload = {
        "target": "price",
        "ensemble_method": cfg["ensemble"]["method"],
        "models": models,
        "holdout_mae": round(cfg.get("metrics", {}).get("mae", 0.0), 3),
        "holdout_rmse": round(cfg.get("metrics", {}).get("rmse", 0.0), 3),
        "pi_coverage": round(cfg.get("pi_coverage", 0.0), 4),
        "last_retrain": issued_at,
        "conformal_quantile": round(cfg.get("conformal_quantile", 0.0), 3),
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(f"Written {METADATA_PATH}")


def write_outputs(
    price_df: pd.DataFrame,
    gen_load_results: dict,
    issued_at: str | None = None,
) -> None:
    """Write all forecast outputs to deploy/data/."""
    issued_at = issued_at or _now_utc()
    write_price_forecast(price_df, issued_at=issued_at)
    _append_price_history(price_df, issued_at=issued_at)
    write_gen_load_forecasts(gen_load_results, issued_at=issued_at)
    write_model_metadata(issued_at=issued_at)
    logger.info("All outputs written to deploy/data/")


def compute_errors(price_df: pd.DataFrame) -> None:
    """Compute forecast errors for yesterday and write to errors/{date}.json.

    Compares yesterday's price forecast (from history) against SMARD actuals.
    Silently skips if actuals or yesterday's forecast are not available yet.
    """
    from datetime import date, timedelta

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    error_path = ERRORS_DIR / f"{yesterday}.json"

    if error_path.exists():
        return  # Already computed

    if not HISTORY_PATH.exists():
        return

    history = json.loads(HISTORY_PATH.read_text())
    yesterday_fc = next(
        (f for f in history.get("forecasts", []) if f.get("issued_at", "")[:10] == yesterday),
        None,
    )
    if yesterday_fc is None:
        return

    # Load SMARD actuals for yesterday
    try:
        from energy_forecasting.config import PROCESSED_DATA_DIR

        merged = pd.read_parquet(PROCESSED_DATA_DIR / "merged.parquet")
        actuals = merged["target_price"].dropna()
        yesterday_actuals = actuals[actuals.index.normalize().astype(str) == yesterday]
        if len(yesterday_actuals) < 24:
            return

        y_pred = np.array([f["forecast"] for f in yesterday_fc["forecasts"]])
        y_true = yesterday_actuals.values[:24]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "date": yesterday,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
        }
        error_path.write_text(json.dumps(payload, indent=2))
        logger.info(f"Daily error for {yesterday}: MAE={mae:.2f}, RMSE={rmse:.2f}")
    except Exception:
        logger.exception(f"Could not compute errors for {yesterday}")
