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

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR
from energy_forecasting.deploy.model_store import (
    ENSEMBLE_CONFIG_PATH,
    load_ensemble_config,
    load_price_model,
    production_model_names,
)

_WEIGHT_THRESHOLD = 1e-10
PRICE_TARGET = "target_price"


def _extend_to_forecast_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Append 24 rows for D+1 delivery hours to the merged dataset.

    The merged dataset is in tz-naive local time (Europe/Berlin delivery hours).
    Tomorrow's delivery hours 0-23 are the rows we need predictions for.
    Features are forward-filled from the last known values; the price target
    column is left as NaN (unknown — this is what we're predicting).

    Returns (extended_df, d1_index) where d1_index identifies the new rows.
    """
    last_date = df.index[-1].date()
    forecast_date = last_date + timedelta(days=1)

    # Check if D+1 rows are already present (e.g., a partial update)
    d1_start = pd.Timestamp(f"{forecast_date} 00:00")
    d1_end = pd.Timestamp(f"{forecast_date} 23:00")
    existing_d1 = df.loc[d1_start:d1_end]
    if len(existing_d1) == 24:
        logger.info(f"D+1 rows already present in merged data for {forecast_date}")
        return df, existing_d1.index

    new_idx = pd.date_range(
        start=d1_start,
        end=d1_end,
        freq="h",
    )
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
    feature_cols = [c for c in extended.columns if c != PRICE_TARGET]
    extended[feature_cols] = extended[feature_cols].ffill()

    logger.info(
        f"Extended merged dataset to {forecast_date}: "
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
    from energy_forecasting.modeling.datasets import DATASET_DIR, find_dataset, load_dataset

    result: dict[str, pd.DataFrame] = {}
    for fv in feature_versions:
        ds_name = _feature_version_to_ds_name(fv)
        # Reuse an existing dataset if it's up-to-date (same number of rows)
        existing = find_dataset(ds_name)
        if existing:
            existing_df = pd.read_parquet(existing)
            # If the dataset already covers the D+1 timestamps, slice directly
            feature_cols = [c for c in existing_df.columns if not c.endswith("__target")]
            target_cols = [c for c in existing_df.columns if c.endswith("__target")]
            if d1_index[-1] in existing_df.index and d1_index[0] in existing_df.index:
                X_d1 = existing_df.loc[d1_index, feature_cols]
                if not X_d1.isna().any(axis=None):
                    logger.info(f"Reusing existing dataset {ds_name} for D+1 features")
                    result[fv] = X_d1
                    continue

        # Fall back to building features from the extended merged dataset.
        # We need the feature list for this version — look it up from the datasets dir
        # or re-engineer from the full config.
        try:
            result[fv] = _engineer_features_for_version(extended_df, d1_index, fv, ds_name)
        except Exception:
            logger.exception(f"Failed to build features for {fv}")

    return result


def _engineer_features_for_version(
    extended_df: pd.DataFrame,
    d1_index: pd.DatetimeIndex,
    feature_version: str,
    ds_name: str,
) -> pd.DataFrame:
    """Build price feature matrix for one feature_version and return D+1 rows."""
    from energy_forecasting.modeling.datasets import DATASET_DIR

    # Load the feature list from the existing dataset columns (most reliable)
    ds_path = DATASET_DIR / f"{ds_name}.parquet"
    if ds_path.exists():
        existing = pd.read_parquet(ds_path)
        feature_cols = [c for c in existing.columns if not c.endswith("__target")]
        # Use extend_features to add the new D+1 rows efficiently
        from energy_forecasting.features.engine import extend_features
        # We need the feature list — recover from the existing dataset columns
        # by matching against the known feature DSL. Since we can't reconstruct the
        # exact DSL strings, we use the columns directly as a pass-through.
        # The extended merged data contains the raw columns; engineer_features
        # re-derives the feature columns.
        # Try the most reliable path: use the full dataset
        from energy_forecasting.config.features import (
            PRICE_FEATURES_FULL,
            PRICE_FEATURES_MAX,
            PRICE_FEATURES_SLIM,
        )
        # Map feature_version back to feature list
        # feature_version names from stage 5c: fs_shap_top90, fs_rfecv_optimum, etc.
        # These are sub-datasets of MAX. The columns in the existing dataset
        # ARE the feature list for this version.
        from energy_forecasting.features.engine import engineer_features as _eng

        # Build full feature matrix and then select only the columns in this version
        full_features = _eng(extended_df, PRICE_FEATURES_MAX, validate=False)
        # Keep only columns that exist in this version's dataset
        available = [c for c in feature_cols if c in full_features.columns]
        X_all = full_features[available]
        X_d1 = X_all.reindex(d1_index)
        logger.info(
            f"Built {fv} features: {len(available)} columns, "
            f"{X_d1.notna().all(axis=1).sum()}/{len(X_d1)} complete D+1 rows"
        )
        return X_d1

    # No existing dataset — build from scratch using MAX features as proxy
    from energy_forecasting.config.features import PRICE_FEATURES_MAX
    from energy_forecasting.features.engine import engineer_features as _eng

    full_features = _eng(extended_df, PRICE_FEATURES_MAX, validate=False)
    X_d1 = full_features.reindex(d1_index)
    logger.warning(
        f"No existing dataset for {feature_version}; using MAX features as proxy "
        f"({len(full_features.columns)} columns)"
    )
    return X_d1


def run_price_inference(
    merged_path: Path | None = None,
    ensemble_config: dict | None = None,
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

    # Extend to D+1
    extended_df, d1_index = _extend_to_forecast_date(df)

    # Identify which feature_versions are needed (non-zero-weight models only)
    prod_names = set(production_model_names(ensemble_config))
    model_entries = {
        e["name"]: e for e in ensemble_config["models"] if e["name"] in prod_names
    }
    feature_versions_needed = {e["feature_version"] for e in model_entries.values()}

    # Build feature matrices
    feature_matrices = _build_feature_matrices(
        extended_df, d1_index, feature_versions_needed
    )

    # Run each production model
    weights = ensemble_config["ensemble"]["weights"]
    predictions: list[np.ndarray] = []
    used_weights: list[float] = []

    for model_name, entry in model_entries.items():
        fv = entry["feature_version"]
        if fv not in feature_matrices:
            logger.warning(f"No feature matrix for {model_name} ({fv}), skipping")
            continue

        X_d1 = feature_matrices[fv]
        if X_d1.isna().any(axis=None):
            n_bad = X_d1.isna().any(axis=1).sum()
            logger.warning(
                f"{model_name}: {n_bad}/{len(X_d1)} rows have NaN features; "
                "may degrade prediction"
            )

        try:
            model = load_price_model(entry["run_id"])
            y_pred = np.asarray(model.predict(X_d1))
            predictions.append(y_pred)
            used_weights.append(weights[model_name])
            logger.debug(
                f"  {model_name}: mean={y_pred.mean():.1f} EUR/MWh "
                f"(weight={weights[model_name]:.3f})"
            )
        except Exception:
            logger.exception(f"Failed prediction for {model_name}")

    if not predictions:
        raise RuntimeError("No price model predictions succeeded; cannot produce forecast")

    # Normalize weights in case any models were skipped
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
    logger.info(
        f"Price inference complete: {len(result)} hours, "
        f"mean={y_blend.mean():.1f} EUR/MWh, "
        f"PI width={conformal_q * 2:.1f} EUR/MWh"
    )
    return result
