"""Write forecast outputs to deploy/data/ for the FastAPI server.

Output structure:
    deploy/data/
    ├── price_forecast.json          ← ForecastResponse for today's price forecast
    ├── gen_load/
    │   ├── wind_onshore_national.json
    │   ├── wind_onshore_50hz.json   (per-TSO)
    │   ├── ...
    │   ├── wind_offshore_national.json
    │   ├── solar_national.json
    │   └── load_national.json
    ├── forecast_history.json        ← rolling 30-day price history
    ├── model_metadata.json          ← ensemble composition
    ├── actuals.json                 ← rolling 30-day price actuals
    ├── gen_load_actuals.json        ← rolling 7-day gen/load actuals
    ├── errors/
    │   └── {date}.json              ← daily price MAE/RMSE once actuals available
    ├── errors_summary.json          ← aggregated price error trend (30 days)
    └── gen_load_errors_summary.json ← per-target gen/load MAE trend (30 days)

All files are loaded by the FastAPI server's dependency layer and served as-is.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_BERLIN_TZ = ZoneInfo("Europe/Berlin")

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config import DEPLOY_DATA_DIR

GEN_LOAD_DATA_DIR = DEPLOY_DATA_DIR / "gen_load"
ERRORS_DIR = DEPLOY_DATA_DIR / "errors"
HISTORY_PATH = DEPLOY_DATA_DIR / "forecast_history.json"
METADATA_PATH = DEPLOY_DATA_DIR / "model_metadata.json"
PRICE_FORECAST_PATH = DEPLOY_DATA_DIR / "price_forecast.json"
HISTORY_DAYS = 30

# TSO region code → per-TSO filename suffix (must match JS GEN_LOAD_CARDS tso keys)
_REGION_SUFFIX: dict[str, str] = {
    "DE_50HZ": "50hz",
    "DE_AMPRION": "amprion",
    "DE_TENNET": "tennet",
    "DE_TRANSNETBW": "transnetbw",
    "DE_CREOS": "creos",
}

# TSO parquet filename stem → column suffix inside that parquet
_TSO_COL_SUFFIX: dict[str, str] = {
    "50Hertz": "_50hz",
    "Amprion": "_ampr",
    "TenneT": "_tenn",
    "TransnetBW": "_tran",
    "Creos": "_lu",
}

# Per-target list of TSO parquets that contribute to national sum.
# Must match GEN_LOAD_TARGETS regions in config/modeling.py.
_TARGET_TSO_FILES: dict[str, list[str]] = {
    "wind_onshore": ["50Hertz", "Amprion", "TenneT", "TransnetBW"],
    "wind_offshore": ["50Hertz", "TenneT"],
    "solar": ["50Hertz", "Amprion", "TenneT", "TransnetBW"],
    "load": ["50Hertz", "Amprion", "TenneT", "TransnetBW", "Creos"],
}

_GEN_LOAD_DISPLAY_TARGETS = {"wind_onshore", "wind_offshore", "solar", "load"}


def _ts(dt) -> str:
    """Format a timestamp as tz-naive ISO 8601 in Europe/Berlin local time.

    Gen/load forecast DataFrames use UTC-aware timestamps (from TSO parquets).
    Converting to Berlin local before stripping tz ensures timestamps in all
    JSON outputs represent delivery hours in German local time, matching the
    actuals format used by write_gen_load_actuals.
    """
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is not None:
            dt = dt.tz_convert(_BERLIN_TZ).tz_localize(None)
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


def write_price_forecast(price_df: pd.DataFrame, issued_at: str | None = None) -> None:
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
    entry: dict = {
        "target": "price",
        "region": "DE_LU",
        "issued_at": issued_at,
        "source": "production",
        "horizon_hours": len(price_df),
        "unit": "EUR/MWh",
        "forecasts": _hourly_entries(price_df),
    }
    model_forecasts = price_df.attrs.get("model_predictions", None)
    if model_forecasts:
        entry["model_forecasts"] = model_forecasts

    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text())
    else:
        history = {"target": "price", "forecasts": [], "count": 0}

    # Deduplicate by delivery date (first forecast timestamp).
    # Using issued_at[:10] was wrong: two runs on different calendar days can both forecast
    # the same delivery date (e.g., an afternoon run after D-day prices are published forecasts D+2
    # while the next morning's run also forecasts D+1 = same delivery date).
    new_delivery = entry["forecasts"][0]["timestamp"][:10] if entry.get("forecasts") else None
    forecasts = [
        f for f in history.get("forecasts", [])
        if not (f.get("forecasts") and f["forecasts"][0]["timestamp"][:10] == new_delivery)
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
    """Write national and per-TSO gen/load forecast JSONs."""
    GEN_LOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    issued_at = issued_at or _now_utc()

    target_to_file = {
        "wind_onshore": "wind_onshore_national.json",
        "wind_offshore": "wind_offshore_national.json",
        "solar": "solar_national.json",
        "load": "load_national.json",
    }
    units = {t: "MW" for t in _GEN_LOAD_DISPLAY_TARGETS}

    # National files
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

    # Per-TSO files — enable the TSO toggles in the dashboard cards
    for (target, region), df in gen_load_results.items():
        if target not in _GEN_LOAD_DISPLAY_TARGETS:
            continue
        suffix = _REGION_SUFFIX.get(region)
        if suffix is None:
            continue
        payload = {
            "target": target,
            "region": region,
            "issued_at": issued_at,
            "horizon_hours": len(df),
            "unit": "MW",
            "forecasts": _hourly_entries(df),
        }
        out = GEN_LOAD_DATA_DIR / f"{target}_{suffix}.json"
        out.write_text(json.dumps(payload, indent=2))
        logger.debug(f"Written {out}")

    # gen_load_diff national — used by dashboard to derive "other generation"
    key_gld = ("gen_load_diff", "DE_NATIONAL")
    if key_gld in gen_load_results:
        df = gen_load_results[key_gld]
        payload = {
            "target": "gen_load_diff",
            "region": "DE_NATIONAL",
            "issued_at": issued_at,
            "horizon_hours": len(df),
            "unit": "MW",
            "forecasts": _hourly_entries(df),
        }
        out = GEN_LOAD_DATA_DIR / "gen_load_diff_national.json"
        out.write_text(json.dumps(payload, indent=2))
        logger.debug(f"Written {out}")

    logger.info("Gen/load forecast JSONs written (national + per-TSO + gen_load_diff)")


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
        models.append(
            {
                "name": entry["name"],
                "category": cat,
                "weight": round(weights.get(entry["name"], 0.0), 6),
                "run_id": entry["run_id"],
                "cv_mae": round(entry.get("config", {}).get("cv_mae", 0.0), 3),
            }
        )

    payload = {
        "target": "price",
        "ensemble_method": cfg["ensemble"]["method"],
        "models": models,
        "holdout_mae": round(cfg.get("metrics", {}).get("mae", 0.0), 3),
        "holdout_rmse": round(cfg.get("metrics", {}).get("rmse", 0.0), 3),
        "pi_coverage": round(cfg.get("pi_coverage", 0.0), 4),
        "last_retrain": issued_at,
        "conformal_quantile": round(cfg.get("conformal_quantile", 0.0), 3),
        "needs_reselection": bool(cfg.get("needs_reselection", False)),
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(f"Written {METADATA_PATH}")


def write_actuals() -> None:
    """Write rolling 30-day actual DE-LU prices to actuals.json."""
    from energy_forecasting.config import PROCESSED_DATA_DIR

    try:
        merged = pd.read_parquet(PROCESSED_DATA_DIR / "merged.parquet", columns=["target_price"])
    except FileNotFoundError:
        return
    # Filter complete days first, then trim to last 30.
    # merged.parquet has exactly 24 rows per day after normalize_dst, so
    # len == 24 only guards against a partial trailing day.
    filtered = merged.dropna(subset=["target_price"])
    by_date = filtered.groupby(filtered.index.date)
    complete = [(d, grp) for d, grp in by_date if len(grp) == 24]
    complete.sort(key=lambda x: x[0])
    days = [
        {"date": str(d), "prices": [round(float(v), 3) for v in grp["target_price"].values]}
        for d, grp in complete[-30:]
    ]
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DATA_DIR / "actuals.json").write_text(
        json.dumps({"days": days, "count": len(days)}, indent=2)
    )
    logger.info(f"Written actuals.json ({len(days)} days)")


def write_gen_load_actuals() -> None:
    """Write rolling 7-day actual gen/load to gen_load_actuals.json.

    Includes the last 7 complete 24h days PLUS today's partial data if available,
    so the actuals overlay connects to the start of the forecast on the dashboard.
    """
    from energy_forecasting.config import PROCESSED_DATA_DIR

    tso_dir = PROCESSED_DATA_DIR / "tso"
    if not tso_dir.exists():
        return

    # Load per-TSO processed parquets
    tso_dfs: dict[str, pd.DataFrame] = {}
    for tso_name in _TSO_COL_SUFFIX:
        path = tso_dir / f"{tso_name}.parquet"
        if path.exists():
            try:
                tso_dfs[tso_name] = pd.read_parquet(path)
            except Exception:
                logger.warning(f"Could not load {path}")

    if not tso_dfs:
        return

    result: dict[str, dict] = {}
    # Also keep Berlin-local combined series to compute gen_load_diff actuals.
    _berlin_combined: dict[str, pd.Series] = {}

    def _to_berlin(s: pd.Series) -> pd.Series:
        idx = s.index
        new_idx = (
            idx.tz_convert("Europe/Berlin").tz_localize(None)
            if idx.tz is not None
            else idx.tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)
        )
        return s.set_axis(new_idx)

    def _series_to_days(combined: pd.Series) -> list[dict]:
        by_date: dict[str, list] = {}
        for ts, val in combined.items():
            if pd.isna(val):
                continue
            by_date.setdefault(ts.strftime("%Y-%m-%d"), []).append(round(float(val), 1))
        sorted_dates = sorted(by_date.items())
        complete = [
            {"date": d, "values": vs}
            for d, vs in sorted_dates
            if len(vs) == 24
        ][-7:]
        if sorted_dates:
            last_date, last_vals = sorted_dates[-1]
            if last_vals and len(last_vals) < 24 and (not complete or complete[-1]["date"] != last_date):
                complete.append({"date": last_date, "values": last_vals})
        return complete

    for target, tsos in _TARGET_TSO_FILES.items():
        series_list = []
        for tso in tsos:
            if tso not in tso_dfs:
                continue
            col = target + _TSO_COL_SUFFIX[tso]
            if col in tso_dfs[tso].columns:
                series_list.append(tso_dfs[tso][col].dropna())
        if not series_list:
            continue

        # Outer join and sum — missing TSO-hours contribute 0
        combined = pd.concat(series_list, axis=1).sum(axis=1, min_count=1)

        # Convert to Berlin local time for delivery-hour date grouping.
        # TSO parquets are UTC; tz-naive parquets are also assumed UTC.
        # Using Berlin local ensures date boundaries match delivery-hour convention
        # and align with the forecast timestamps written by _ts().
        combined = _to_berlin(combined)
        _berlin_combined[target] = combined

        complete = _series_to_days(combined)
        if complete:
            result[target] = {"days": complete}

    # gen_load_diff actuals = load − wind_onshore − wind_offshore − solar
    # Represents dispatchable/other generation; may be negative during high renewables.
    _renewables = ["wind_onshore", "wind_offshore", "solar"]
    if "load" in _berlin_combined and all(t in _berlin_combined for t in _renewables):
        ref = _berlin_combined["load"]
        gld = ref.copy()
        for t in _renewables:
            gld = gld.sub(_berlin_combined[t].reindex(ref.index, fill_value=0))
        complete = _series_to_days(gld)
        if complete:
            result["gen_load_diff"] = {"days": complete}

    if result:
        DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DEPLOY_DATA_DIR / "gen_load_actuals.json").write_text(json.dumps(result, indent=2))
        n = sum(len(v["days"]) for v in result.values())
        logger.info(f"Written gen_load_actuals.json ({n} total target-days)")


def write_errors_summary() -> None:
    """Aggregate errors/*.json into errors_summary.json (last 30 days)."""
    if not ERRORS_DIR.exists():
        return
    records = sorted(
        [
            json.loads(p.read_text())
            for p in ERRORS_DIR.glob("*.json")
            if p.stem[0].isdigit()
        ],
        key=lambda r: r["date"],
    )[-30:]
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DATA_DIR / "errors_summary.json").write_text(
        json.dumps(
            {
                "dates": [r["date"] for r in records],
                "mae": [r["mae"] for r in records],
                "rmse": [r["rmse"] for r in records],
            },
            indent=2,
        )
    )
    logger.info(f"Written errors_summary.json ({len(records)} entries)")


def write_gen_load_errors() -> None:
    """Compute daily gen/load MAE vs SMARD actuals; write gen_load_errors_summary.json.

    Requires historical_forecasts/*.parquet to have accumulated across runs (preserved
    by the CI deploy-state cache).  Silently no-ops if data is insufficient.
    """
    from energy_forecasting.config import HISTORICAL_FORECASTS_DIR, PROCESSED_DATA_DIR

    tso_dir = PROCESSED_DATA_DIR / "tso"
    if not tso_dir.exists():
        return

    # Load TSO actuals
    tso_dfs: dict[str, pd.DataFrame] = {}
    for tso_name in _TSO_COL_SUFFIX:
        path = tso_dir / f"{tso_name}.parquet"
        if path.exists():
            try:
                tso_dfs[tso_name] = pd.read_parquet(path)
            except Exception:
                pass

    if not tso_dfs:
        return

    result: dict[str, dict] = {}
    for target, tsos in _TARGET_TSO_FILES.items():
        # Build national actual series
        series_list = []
        for tso in tsos:
            if tso not in tso_dfs:
                continue
            col = target + _TSO_COL_SUFFIX[tso]
            if col in tso_dfs[tso].columns:
                series_list.append(tso_dfs[tso][col].dropna())
        if not series_list:
            continue
        actuals = pd.concat(series_list, axis=1).sum(axis=1, min_count=1)
        idx = actuals.index
        if hasattr(idx, "tz") and idx.tz is not None:
            actuals.index = idx.tz_convert("UTC")
        else:
            actuals.index = idx.tz_localize("UTC")

        # Load historical forecasts for the national target
        hf_path = HISTORICAL_FORECASTS_DIR / f"{target}_DE_NATIONAL.parquet"
        if not hf_path.exists():
            continue
        try:
            hf = pd.read_parquet(hf_path)[["y_pred"]]
        except Exception:
            continue

        hf_idx = hf.index
        if hasattr(hf_idx, "tz") and hf_idx.tz is not None:
            hf.index = hf_idx.tz_convert("UTC")
        else:
            hf.index = hf_idx.tz_localize("UTC")

        # Join forecasts with actuals on UTC timestamps
        df_both = hf.copy()
        df_both["y_true"] = actuals.reindex(hf.index)
        df_both = df_both.dropna()
        if df_both.empty:
            continue

        df_both["date"] = [str(ts)[:10] for ts in df_both.index.tz_convert("UTC")]
        daily = (
            df_both.groupby("date")
            .apply(
                lambda g: pd.Series(
                    {
                        "mae": float(np.mean(np.abs(g["y_pred"] - g["y_true"]))),
                        "n": len(g),
                    }
                )
            )
            .query("n == 24")
            .drop(columns="n")
            .tail(30)
        )
        if daily.empty:
            continue
        result[target] = {
            "dates": list(daily.index),
            "mae": [round(v, 1) for v in daily["mae"]],
        }

    if result:
        DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DEPLOY_DATA_DIR / "gen_load_errors_summary.json").write_text(
            json.dumps(result, indent=2)
        )
        logger.info(f"Written gen_load_errors_summary.json ({len(result)} targets)")


def write_gen_load_hindcast() -> None:
    """Write last 7 days of national gen/load model predictions to gen_load_hindcast.json.

    Reads y_pred from the historical_forecasts parquets (which accumulate both OOF
    and production predictions), so the dashboard can overlay the model's past
    predictions against SMARD actuals for a visual accuracy comparison.
    """
    from energy_forecasting.config import HISTORICAL_FORECASTS_DIR

    targets = ["wind_onshore", "wind_offshore", "solar", "load", "gen_load_diff"]
    result: dict[str, list] = {}

    for target in targets:
        hf_path = HISTORICAL_FORECASTS_DIR / f"{target}_DE_NATIONAL.parquet"
        if not hf_path.exists():
            continue
        try:
            hf = pd.read_parquet(hf_path)[["y_pred"]]
        except Exception:
            continue

        idx = hf.index
        if hasattr(idx, "tz") and idx.tz is not None:
            hf.index = idx.tz_convert("UTC")
        else:
            hf.index = idx.tz_localize("UTC")

        # Past 7 days: exclude future predictions (index < now) so we only
        # show delivery times that have already passed and have real actuals.
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - pd.Timedelta(days=7)
        hf = hf[(hf.index > cutoff) & (hf.index < now)].dropna()
        if hf.empty:
            continue

        result[target] = [
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "forecast": round(float(val), 1),
            }
            for ts, val in hf["y_pred"].items()
        ]

    if result:
        DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DEPLOY_DATA_DIR / "gen_load_hindcast.json").write_text(json.dumps(result, indent=2))
        logger.info(f"Written gen_load_hindcast.json ({len(result)} targets)")


def backfill_price_history_from_model(n_days: int = 60) -> int:
    """Retroactively generate price forecasts for the past n_days using the production ensemble.

    Runs feature engineering ONCE on merged.parquet, then batch-predicts for all target
    delivery dates not already covered by production entries in forecast_history.json.
    Injects pseudo-history entries (source="backtest") so the price history and error
    charts can display 30+ days of data before production runs have accumulated.

    Returns the number of new pseudo-history entries written.
    """
    try:
        from energy_forecasting.config import PROCESSED_DATA_DIR
        from energy_forecasting.config.features import PRICE_FEATURES_MAX
        from energy_forecasting.features.engine import engineer_features as _eng
        from energy_forecasting.modeling.datasets import DATASET_DIR
        from energy_forecasting.deploy.model_store import (
            load_ensemble_config,
            load_price_model,
            load_price_model_scaler,
            production_model_names,
        )
    except ImportError as exc:
        logger.warning(f"backfill_price_history_from_model: import failed — {exc}")
        return 0

    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text())
    else:
        history = {"target": "price", "forecasts": [], "count": 0}

    existing_delivery_dates: set[str] = {
        e["forecasts"][0]["timestamp"][:10]
        for e in history.get("forecasts", [])
        if e.get("forecasts") and len(e["forecasts"]) >= 24
    }

    merged_path = PROCESSED_DATA_DIR / "merged.parquet"
    if not merged_path.exists():
        logger.warning("backfill_price_history_from_model: merged.parquet not found")
        return 0

    merged = pd.read_parquet(merged_path)
    last_ts = merged.index.max()

    target_dates = []
    for i in range(n_days, 0, -1):
        d = (last_ts - pd.Timedelta(days=i)).normalize()
        delivery_date = d.strftime("%Y-%m-%d")
        if delivery_date not in existing_delivery_dates:
            target_dates.append(delivery_date)

    if not target_dates:
        logger.info("backfill_price_history_from_model: all dates already covered")
        return 0

    logger.info(f"backfill_price_history_from_model: engineering features for {len(target_dates)} target dates…")
    try:
        full_features = _eng(merged, PRICE_FEATURES_MAX, validate=False)
    except Exception:
        logger.exception("backfill_price_history_from_model: feature engineering failed")
        return 0

    try:
        cfg = load_ensemble_config()
        prod_names = set(production_model_names(cfg))
        model_entries = {e["name"]: e for e in cfg["models"] if e["name"] in prod_names}
        weights = cfg["ensemble"]["weights"]
    except Exception:
        logger.exception("backfill_price_history_from_model: failed to load ensemble config")
        return 0

    # Load feature column lists per version.
    # Priority: (1) existing dataset parquets via pyarrow schema (fast, no full read),
    # (2) price_feature_cols.json downloaded from Release alongside model artifacts.
    _feature_cols_cache: dict[str, list[str]] | None = None
    def _load_feature_cols_json() -> dict[str, list[str]]:
        nonlocal _feature_cols_cache
        if _feature_cols_cache is not None:
            return _feature_cols_cache
        # Look in models/ dir (downloaded alongside model artifacts in CI)
        from energy_forecasting.config import MODELS_DIR
        candidate = MODELS_DIR / "price" / ".." / "price_feature_cols.json"
        # Normalise: models/price/../price_feature_cols.json → models/price_feature_cols.json
        candidate = (MODELS_DIR / "price_feature_cols.json").resolve()
        if candidate.exists():
            _feature_cols_cache = json.loads(candidate.read_text())
        else:
            _feature_cols_cache = {}
        return _feature_cols_cache

    version_cols: dict[str, list[str]] = {}
    for entry in model_entries.values():
        fv = entry["feature_version"]
        if fv in version_cols:
            continue
        ds_path = DATASET_DIR / f"price_{fv}.parquet"
        if ds_path.exists():
            try:
                import pyarrow.parquet as _pq
                schema = _pq.read_schema(ds_path)
                fcols = [c for c in schema.names if not c.endswith("__target")]
            except Exception:
                ds_sample = pd.read_parquet(ds_path)
                fcols = [c for c in ds_sample.columns if not c.endswith("__target")]
            version_cols[fv] = [c for c in fcols if c in full_features.columns]
        else:
            cached = _load_feature_cols_json().get(fv)
            if cached:
                version_cols[fv] = [c for c in cached if c in full_features.columns]
            else:
                logger.warning(f"backfill_price_history_from_model: no column list for {fv}; skipping")
                version_cols[fv] = []

    loaded: dict[str, tuple] = {}
    for name, entry in model_entries.items():
        try:
            model = load_price_model(entry["run_id"])
            scaler = load_price_model_scaler(entry["run_id"])
            loaded[name] = (model, scaler, entry["feature_version"])
        except Exception:
            logger.warning(f"backfill_price_history_from_model: could not load model {name}")

    if not loaded:
        logger.warning("backfill_price_history_from_model: no models loaded")
        return 0

    new_entries: list[dict] = []
    for delivery_date in target_dates:
        day_start = pd.Timestamp(f"{delivery_date} 00:00")
        day_end = pd.Timestamp(f"{delivery_date} 23:00")
        if day_start not in full_features.index or day_end not in full_features.index:
            continue

        preds: list[np.ndarray] = []
        w_used: list[float] = []
        model_preds_by_name: dict[str, list[float]] = {}
        skip = False
        for name, (model, scaler, fv) in loaded.items():
            cols = version_cols.get(fv, [])
            if not cols:
                skip = True
                break
            try:
                X = full_features.loc[day_start:day_end, cols]
            except Exception:
                skip = True
                break
            if X.shape[0] != 24 or X.isna().any(axis=None):
                skip = True
                break
            X_arr = scaler.transform(X) if scaler is not None else X.values
            try:
                y = np.asarray(model.predict(X_arr))
                model_preds_by_name[name] = y.tolist()
                preds.append(y)
                w_used.append(weights[name])
            except Exception:
                skip = True
                break
        if skip or not preds:
            continue

        w = np.array(w_used, dtype=float)
        w /= w.sum()
        y_blend = (np.stack(preds, axis=1) * w).sum(axis=1)

        issued_at = (
            pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%dT08:00:00Z")
        new_entry: dict = {
            "target": "price",
            "region": "DE_LU",
            "issued_at": issued_at,
            "horizon_hours": 24,
            "unit": "EUR/MWh",
            "source": "backtest",
            "forecasts": [
                {"timestamp": f"{delivery_date}T{h:02d}:00:00", "forecast": round(float(y_blend[h]), 3)}
                for h in range(24)
            ],
        }
        if model_preds_by_name:
            new_entry["model_forecasts"] = model_preds_by_name
        new_entries.append(new_entry)

    if not new_entries:
        logger.info("backfill_price_history_from_model: no entries generated (feature gaps or model errors)")
        return 0

    all_forecasts = new_entries + history.get("forecasts", [])
    all_forecasts.sort(key=lambda e: e.get("issued_at", ""))
    all_forecasts = all_forecasts[-HISTORY_DAYS:]
    history["forecasts"] = all_forecasts
    history["count"] = len(all_forecasts)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    logger.info(f"backfill_price_history_from_model: added {len(new_entries)} pseudo-history entries")
    return len(new_entries)


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
    write_actuals()
    write_gen_load_actuals()
    backfill_errors()
    write_gen_load_errors()
    write_gen_load_hindcast()
    write_errors_summary()
    logger.info("All outputs written to deploy/data/")


def backfill_errors() -> None:
    """Compute MAE/RMSE for every forecast in history where actuals are available.

    Scans forecast_history.json, matches each entry to its delivery date (first
    forecast timestamp date), and writes errors/{delivery_date}.json if missing.
    This replaces the old single-day compute_errors approach and correctly handles
    the case where the pipeline runs after noon (D+2 forecast) as well as the
    standard 08:00 UTC case (D+1 forecast).
    """
    if not HISTORY_PATH.exists():
        return

    history = json.loads(HISTORY_PATH.read_text())
    fc_entries = history.get("forecasts", [])
    if not fc_entries:
        return

    try:
        from energy_forecasting.config import PROCESSED_DATA_DIR

        merged = pd.read_parquet(PROCESSED_DATA_DIR / "merged.parquet")
        actuals = merged["target_price"].dropna()
    except FileNotFoundError:
        return

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in fc_entries:
        fc = entry.get("forecasts", [])
        if len(fc) < 24:
            continue
        # Match by delivery date (first forecast timestamp), not issued_at date.
        # Standard runs forecast D+1, so issued_at[:10] differs from delivery date.
        delivery_date = fc[0]["timestamp"][:10]
        error_path = ERRORS_DIR / f"{delivery_date}.json"
        source = entry.get("source", "production")
        issued_at = entry.get("issued_at")
        if error_path.exists():
            try:
                existing = json.loads(error_path.read_text())
            except json.JSONDecodeError:
                existing = {}
            if (
                existing.get("source") == source
                and existing.get("issued_at") == issued_at
            ):
                continue

        day_actuals = actuals[actuals.index.normalize().astype(str) == delivery_date]
        if len(day_actuals) < 24:
            continue  # Actuals not yet published for this delivery date

        y_pred = np.array([f["forecast"] for f in fc[:24]])
        y_true = day_actuals.values[:24]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        payload = {
            "date": delivery_date,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "source": source,
            "issued_at": issued_at,
        }
        error_path.write_text(json.dumps(payload, indent=2))
        logger.info(f"Backfilled error for {delivery_date}: MAE={mae:.2f}, RMSE={rmse:.2f}")
