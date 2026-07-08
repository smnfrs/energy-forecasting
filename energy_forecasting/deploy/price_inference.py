"""Daily price inference pipeline.

Produces a 24h D+1 price forecast using the production SLSQP ensemble.

Steps:
1. Load ensemble_config.json
2. Load merged.parquet and apply EMA overlay (uses live gen/load forecasts)
3. Extend dataset to D+1 delivery hours (forward-fill known features)
4. Compute price feature matrix for each unique feature_version
5. Load each non-zero-weight base model and predict 24 D+1 hours
6. Apply SLSQP ensemble weights → blend forecast
7. Apply conformal PI calibration from ensemble_config

The output is a DataFrame with 24 rows and columns [y_pred, y_lower, y_upper]
indexed by D+1 delivery hour timestamps (local time, tz-naive to match merged.parquet).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR
from energy_forecasting.deploy.model_store import (
    load_ensemble_config,
    load_price_model,
    load_price_model_scaler,
    production_model_names,
)

_WEIGHT_THRESHOLD = 1e-10
PRICE_TARGET = "target_price"
LOCAL_TZ = ZoneInfo("Europe/Berlin")


def _default_delivery_date(now: datetime | None = None) -> date:
    """Return the next Berlin delivery date for the daily price forecast."""
    current = now or datetime.now(LOCAL_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=LOCAL_TZ)
    else:
        current = current.astimezone(LOCAL_TZ)
    return current.date() + timedelta(days=1)


def _extend_to_forecast_date(
    df: pd.DataFrame,
    forecast_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Ensure 24 delivery-hour rows exist for the requested forecast date.

    The merged dataset is in tz-naive local time (Europe/Berlin delivery hours).
    Tomorrow's delivery hours 0-23 are the rows we need predictions for by
    default. The date is based on Berlin wall-clock time, not on the last row in
    merged.parquet: after the market clears, merged data may already contain
    tomorrow's target prices, but the daily forecast should still be D+1.

    Features are forward-filled from the last known values; the price target
    column is left as NaN (unknown — this is what we're predicting).

    Returns (extended_df, forecast_index) where forecast_index identifies the
    24 delivery rows.
    """
    forecast_date = forecast_date or _default_delivery_date()

    d1_start = pd.Timestamp(f"{forecast_date} 00:00")
    d1_end = pd.Timestamp(f"{forecast_date} 23:00")
    existing_d1 = df.loc[d1_start:d1_end]
    new_idx = pd.date_range(
        start=d1_start,
        end=d1_end,
        freq="h",
    )

    if len(existing_d1) == 24:
        extended = df.copy()
        logger.info(f"Delivery rows already present in merged data for {forecast_date}")
    else:
        new_rows = pd.DataFrame(np.nan, index=new_idx, columns=df.columns)
        extended = pd.concat([df, new_rows])

    extended = extended[~extended.index.duplicated(keep="last")]

    # Forward-fill all feature columns for D+1 rows.
    # - Commodity prices (daily data): ffill propagates yesterday's close
    # - Temporal indicators (regime, etc.): ffill from D-1 as placeholder;
    #   the actual temporal feature values are computed at engineer_features time
    # - SMARD day-ahead generation forecasts (prognostizierte_*): SMARD publishes
    #   these for D+1 around 18:00 CET D-1; by 08:00 UTC they're already in
    #   the raw data from the data update step. If not yet available, ffill.
    # Target column stays NaN so engineer_features drops it for the D+1 rows
    # if 'target_price__target' is used.
    original_before_fill = extended.copy()
    feature_cols = [c for c in extended.columns if c != PRICE_TARGET]
    extended[feature_cols] = extended[feature_cols].ffill()
    if PRICE_TARGET in extended.columns:
        extended.loc[new_idx, PRICE_TARGET] = np.nan
    from energy_forecasting.deploy.feature_monitoring import summarize_source_availability

    extended.attrs["source_availability"] = summarize_source_availability(
        original_before_fill,
        new_idx,
        exclude_columns={PRICE_TARGET},
    )

    logger.info(
        f"Prepared merged dataset for delivery date {forecast_date}: "
        f"{len(df)} → {len(extended)} rows"
    )
    return extended, new_idx


def _feature_version_to_ds_name(feature_version: str) -> str:
    """Map feature_version from ensemble_config to a dataset name."""
    # feature_version values like 'fs_shap_top90', 'fs_rfecv_optimum', etc.
    return f"price_{feature_version}"


def _build_feature_matrices(
    extended_df: pd.DataFrame,
    d1_index: pd.DatetimeIndex,
    feature_versions: set[str],
) -> dict[str, pd.DataFrame]:
    """Compute price feature matrices for each unique feature_version.

    Runs engineer_features on the full extended dataset, then slices to D+1.
    This ensures rolling/lag features are correctly computed using the full
    historical series.

    Returns dict of {feature_version: X_d1} where X_d1 is a 24-row DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for fv in feature_versions:
        ds_name = _feature_version_to_ds_name(fv)
        result[fv] = _engineer_features_for_version(extended_df, d1_index, fv, ds_name)

    return result


def _trained_feature_columns(feature_version: str, ds_name: str) -> list[str]:
    """Load the exact feature columns used by a trained price model."""
    from energy_forecasting.modeling.datasets import DATASET_DIR

    ds_path = DATASET_DIR / f"{ds_name}.parquet"
    if ds_path.exists():
        try:
            import pyarrow.parquet as pq

            names = pq.read_schema(ds_path).names
        except Exception:
            names = list(pd.read_parquet(ds_path).columns)
        return [c for c in names if not c.endswith("__target") and c != "__index_level_0__"]

    cols_path = MODELS_DIR / "price_feature_cols.json"
    if cols_path.exists():
        all_cols = json.loads(cols_path.read_text())
        if feature_version in all_cols:
            return [c for c in all_cols[feature_version] if c != "__index_level_0__"]

    raise FileNotFoundError(
        f"No trained feature-column list found for {feature_version}. "
        f"Expected {ds_path} or {cols_path}."
    )


def _engineer_features_for_version(
    extended_df: pd.DataFrame,
    d1_index: pd.DatetimeIndex,
    feature_version: str,
    ds_name: str,
) -> pd.DataFrame:
    """Build price feature matrix for one feature_version and return D+1 rows."""
    from energy_forecasting.config.features import PRICE_FEATURES_MAX
    from energy_forecasting.features.engine import engineer_features as _eng

    feature_cols = _trained_feature_columns(feature_version, ds_name)
    full_features = _eng(extended_df, PRICE_FEATURES_MAX, validate=False)
    missing = [c for c in feature_cols if c not in full_features.columns]
    if missing:
        raise KeyError(
            f"{feature_version}: {len(missing)} trained feature column(s) were not "
            f"recomputed for inference: {missing[:10]}"
        )

    X_all = full_features[feature_cols]
    X_d1 = X_all.reindex(d1_index)
    from energy_forecasting.deploy.feature_monitoring import dataframe_records, summarize_matrix

    X_d1.attrs["feature_audit"] = {
        "feature_version": feature_version,
        "dataset_name": ds_name,
        "configured_feature_count": len(feature_cols),
        "engineered_feature_count": int(full_features.shape[1]),
        "configured_features": feature_cols,
        "missing_configured_features": missing,
        "matrix": summarize_matrix(X_d1, timestamps=d1_index),
        "feature_values": dataframe_records(X_d1),
    }
    logger.info(
        f"Built {feature_version} features: {len(feature_cols)} columns, "
        f"{X_d1.notna().all(axis=1).sum()}/{len(X_d1)} complete D+1 rows"
    )
    return X_d1


def run_price_inference(
    merged_path: Path | None = None,
    ensemble_config: dict | None = None,
    forecast_date: date | None = None,
) -> pd.DataFrame:
    """Produce a 24h D+1 price forecast using the production SLSQP ensemble.

    Returns a 24-row DataFrame with columns [y_pred, y_lower, y_upper] indexed
    by D+1 delivery hour timestamps (tz-naive local time).
    """
    if ensemble_config is None:
        ensemble_config = load_ensemble_config()

    merged_path = merged_path or PROCESSED_DATA_DIR / "merged.parquet"
    df = pd.read_parquet(merged_path)
    logger.info(f"Loaded merged data: {df.shape}, last={df.index[-1]}")

    # Apply EMA overlay — uses today's live gen/load forecasts from
    # historical_forecasts/*.parquet to update prog_* columns
    from energy_forecasting.modeling.price import _overlay_ema_forecasts

    df = _overlay_ema_forecasts(df)

    # Extend to the intended D+1 delivery date.
    extended_df, d1_index = _extend_to_forecast_date(df, forecast_date=forecast_date)

    # Identify which feature_versions are needed (non-zero-weight models only)
    prod_names = set(production_model_names(ensemble_config))
    model_entries = {e["name"]: e for e in ensemble_config["models"] if e["name"] in prod_names}
    if not model_entries:
        raise RuntimeError("No production price models configured")
    feature_versions_needed = {e["feature_version"] for e in model_entries.values()}

    # Build feature matrices
    feature_matrices = _build_feature_matrices(extended_df, d1_index, feature_versions_needed)
    feature_audit_by_version = {
        fv: matrix.attrs.get("feature_audit", {})
        for fv, matrix in feature_matrices.items()
    }

    # Run each production model
    weights = ensemble_config["ensemble"]["weights"]
    predictions: list[np.ndarray] = []
    used_weights: list[float] = []
    model_predictions: dict[str, list[float]] = {}

    for model_name, entry in model_entries.items():
        fv = entry["feature_version"]
        if fv not in feature_matrices:
            raise RuntimeError(f"No feature matrix for production model {model_name} ({fv})")

        X_d1 = feature_matrices[fv]
        if X_d1.isna().any(axis=None):
            bad_cols = X_d1.columns[X_d1.isna().any()].tolist()
            n_bad = int(X_d1.isna().any(axis=1).sum())
            raise RuntimeError(
                f"{model_name}: {n_bad}/{len(X_d1)} forecast rows have NaN features; "
                f"first bad columns: {bad_cols[:10]}"
            )

        try:
            model = load_price_model(entry["run_id"])
            scaler = load_price_model_scaler(entry["run_id"])
            X_to_predict = scaler.transform(X_d1) if scaler is not None else X_d1
            y_pred = np.asarray(model.predict(X_to_predict))
        except Exception as exc:
            raise RuntimeError(f"Failed prediction for production model {model_name}") from exc

        model_predictions[model_name] = y_pred.tolist()
        predictions.append(y_pred)
        used_weights.append(weights[model_name])
        logger.debug(
            f"  {model_name}: mean={y_pred.mean():.1f} EUR/MWh "
            f"(weight={weights[model_name]:.3f})"
        )

    if len(predictions) != len(model_entries):
        raise RuntimeError(
            f"Only {len(predictions)}/{len(model_entries)} production price models produced predictions"
        )

    # Normalize configured production weights. All production models must have succeeded.
    w = np.array(used_weights)
    w = w / w.sum()

    # Weighted blend
    pred_matrix = np.stack(predictions, axis=1)  # (24, n_models)
    y_blend = (pred_matrix * w).sum(axis=1)

    # Conformal prediction interval (symmetric, calibrated on holdout)
    conformal_q = float(ensemble_config.get("conformal_quantile", 0.0))
    y_lower = y_blend - conformal_q
    y_upper = y_blend + conformal_q

    result = pd.DataFrame(
        {
            "y_pred": y_blend,
            "y_lower": y_lower,
            "y_upper": y_upper,
        },
        index=d1_index,
    )
    result.attrs["model_predictions"] = model_predictions
    result.attrs["feature_audit"] = {
        "target": "price",
        "region": "DE_LU",
        "forecast_start": d1_index[0].isoformat(),
        "forecast_end": d1_index[-1].isoformat(),
        "source_availability": extended_df.attrs.get("source_availability", {}),
        "feature_versions": [
            feature_audit_by_version[fv]
            for fv in sorted(feature_audit_by_version)
        ],
        "models": [
            {
                "name": name,
                "run_id": entry["run_id"],
                "feature_version": entry["feature_version"],
                "weight": float(weights.get(name, 0.0)),
            }
            for name, entry in sorted(model_entries.items())
        ],
    }
    logger.info(
        f"Price inference complete: {len(result)} hours, "
        f"mean={y_blend.mean():.1f} EUR/MWh, "
        f"PI width={conformal_q * 2:.1f} EUR/MWh"
    )
    return result
