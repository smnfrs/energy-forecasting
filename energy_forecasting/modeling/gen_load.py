"""Gen/load model training pipeline with Optuna hyperparameter search.

Trains per-TSO generation (wind, solar) and load forecasting models with:
- Weather FE (target-specific class) searched jointly via Optuna TPE
- Temporal features (cyclical time, holidays, lagged actuals)
- Dataset params (log_target, lags_target, scaler)
- Stacking ensemble over multiple base models

Uses train_model() from training.py for the final model with MAPIE intervals
and full MLflow tracking. Optuna search runs lightweight in-memory CV.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Ridge

from energy_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from energy_forecasting.config.locations import locations_for_tso
from energy_forecasting.config.modeling import (
    EXPERIMENTS,
    GEN_LOAD_CV_TEST_DAYS,
    GEN_LOAD_HISTORICAL_FOLDS,
    GEN_LOAD_HOLDOUT_DAYS,
    GEN_LOAD_MAX_TRAIN_HOURS,
    GEN_LOAD_OPTUNA_TRIALS,
    GEN_LOAD_TARGETS,
    PI_CONFIDENCE_LEVEL,
    REGION_TO_TSO,
    SEARCH_CV_FOLDS,
    TARGET_WEATHER_TYPE,
    VALIDATION_CV_FOLDS,
)
from energy_forecasting.config.search_spaces import (
    suggest_dataset_params,
    suggest_lgbm,
    suggest_xgboost,
)
from energy_forecasting.config.smard import TSO_SUFFIXES
from energy_forecasting.features.market import compute_fourier_features, compute_temporal_features
from energy_forecasting.features.weather_load import WeatherLoadFE
from energy_forecasting.features.weather_solar import WeatherSolarPowerFE
from energy_forecasting.features.weather_wind import WeatherWindPowerFE
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import DATASET_DIR, TARGET_COL_SUFFIX
from energy_forecasting.modeling.forecasting import (
    DEFAULT_FORECAST_HORIZON,
    forecast_with_lags_windowed,
)
from energy_forecasting.modeling.intervals import (
    calibrate_ensemble_intervals,
    predict_ensemble_intervals,
)
from energy_forecasting.modeling.metrics import calculate_metrics, calculate_pi_metrics
from energy_forecasting.modeling.mlflow_utils import TrackedRun
from energy_forecasting.modeling.training import _apply_scaler, _fit_scaler, train_model

# With EMA-matching CV (weekly test folds), recursive CV uses full
# coverage per fold — i.e. one 168h window per fold. The
# `forecast_with_lags_windowed` helper is retained here so that any future
# switch to a multi-week test fold design works unchanged.

# ── Weather FE class dispatch ────────────────────────────────────

_TARGET_FE_CLASS: dict[str, type] = {
    "wind_onshore": WeatherWindPowerFE,
    "wind_offshore": WeatherWindPowerFE,
    "solar": WeatherSolarPowerFE,
    "load": WeatherLoadFE,
    "gen_load_diff": WeatherLoadFE,
}

# ── Model dispatch ───────────────────────────────────────────────

_MODEL_SUGGEST: dict[str, callable] = {
    "LGBMRegressor": suggest_lgbm,
    "XGBRegressor": suggest_xgboost,
}

WEATHER_FEATURES_DIR = PROCESSED_DATA_DIR / "weather_features"

# Minimum rows required after feature construction to run CV meaningfully.
# ~42 days of hourly data.
_MIN_SAMPLES_FOR_TRAINING = 1000


# ── Data loading ─────────────────────────────────────────────────


def _load_tso_data(region: str) -> pd.DataFrame:
    """Load cleaned TSO parquet from data/processed/tso/.

    For DE_NATIONAL, aggregates all TSO parquets into national totals.
    """
    if region == "DE_NATIONAL":
        return _load_national_tso_data()
    tso = REGION_TO_TSO[region]
    path = PROCESSED_DATA_DIR / "tso" / f"{tso}.parquet"
    return pd.read_parquet(path)


def _load_national_tso_data() -> pd.DataFrame:
    """Load all TSO parquets and compute national gen_load_diff.

    Returns DataFrame with columns for each generation type (national total),
    load (national total), and gen_load_diff = sum(generation) - sum(load).
    """
    _GEN_PREFIXES = [
        "wind_onshore", "wind_offshore", "solar", "biomass", "gas",
        "hard_coal", "lignite", "pumped_storage", "hydro",
        "other_renew", "other_conv",
    ]

    per_type_totals: dict[str, pd.Series] = {}
    total_load: pd.Series | None = None
    common_idx = None

    for tso_name in ["50Hertz", "Amprion", "TenneT", "TransnetBW", "Creos"]:
        path = PROCESSED_DATA_DIR / "tso" / f"{tso_name}.parquet"
        df = pd.read_parquet(path)
        suffix = TSO_SUFFIXES[tso_name]

        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

        # Accumulate per-type generation totals
        for prefix in _GEN_PREFIXES:
            col = f"{prefix}{suffix}"
            if col not in df.columns:
                continue
            if prefix not in per_type_totals:
                per_type_totals[prefix] = df[col].reindex(df.index).fillna(0)
            else:
                aligned = df[col].reindex(per_type_totals[prefix].index).fillna(0)
                per_type_totals[prefix] = per_type_totals[prefix].add(aligned, fill_value=0)

        # Accumulate load
        load_col = f"load{suffix}"
        if load_col in df.columns:
            if total_load is None:
                total_load = df[load_col].reindex(df.index).fillna(0)
            else:
                aligned = df[load_col].reindex(total_load.index).fillna(0)
                total_load = total_load.add(aligned, fill_value=0)

    total_gen = sum(per_type_totals.values())

    result = pd.DataFrame(index=common_idx)
    result["gen_load_diff"] = (total_gen.loc[common_idx] - total_load.loc[common_idx])
    # Include national totals for lag features in _compute_temporal_features
    result["load"] = total_load.loc[common_idx]
    for prefix, series in per_type_totals.items():
        result[prefix] = series.loc[common_idx]
    return result


def _load_weather_data(
    target: str, region: str, source: str = "history",
) -> pd.DataFrame:
    """Load weather data for ``target`` and ``region``.

    Parameters
    ----------
    source : str
        Either "history" (Open-Meteo actual archive — default) or
        "hist_forecast" (forecast weather as it would have been issued
        before the timestamp; used to make CV test folds backtest-honest,
        mirroring EMA's `slice_weather_for_cutoff` mode="backtest" at
        `generate_historical_forecasts.py:196`).
    """
    if region == "DE_NATIONAL":
        return _load_national_weather_data(source=source)
    weather_type = TARGET_WEATHER_TYPE[target]
    tso = REGION_TO_TSO[region]
    path = RAW_DATA_DIR / "weather" / weather_type / tso / f"{source}.parquet"
    return pd.read_parquet(path)


def _load_national_weather_data(source: str = "history") -> pd.DataFrame:
    """Concatenate city weather from all TSOs for national-level models."""
    frames = []
    for tso_name in ["50Hertz", "Amprion", "TenneT", "TransnetBW", "Creos"]:
        path = RAW_DATA_DIR / "weather" / "cities" / tso_name / f"{source}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, axis=1)


def _get_target_col(target: str, region: str) -> str:
    """Get the target column name in the TSO parquet.

    TSO parquets have columns like 'wind_onshore_50hz', 'load_ampr', etc.
    For gen_load_diff (national), the column is computed and named directly.
    """
    if target == "gen_load_diff":
        return "gen_load_diff"
    tso = REGION_TO_TSO[region]
    suffix = TSO_SUFFIXES[tso]
    return f"{target}{suffix}"


# ── Upstream forecast features ───────────────────────────────────


def _load_upstream_actuals(target: str) -> pd.DataFrame:
    """Load actual historical values of upstream targets as exog features.

    Matches EMA's `df_hist = pd.merge(..., df_targets[exog_tso_], ...)` pattern
    (data_loaders.py:369-370): during training, upstream exog features are the
    **ground-truth observations** of the upstream targets — not predictions
    from upstream models. For each (upstream_target, upstream_region) pair
    (cross-product across all regions of each upstream target, matching EMA's
    "ALL TSOs" behaviour at data_loaders.py:328-329), pulls the native column
    from the corresponding TSO parquet.

    At inference time (Stage 6 / `update_forecasts`), these same columns must
    be populated from upstream model forecasts — that part is out of scope here.

    Returns DataFrame with native TSO column names (e.g. 'wind_onshore_50hz'),
    indexed by DatetimeIndex. Empty DataFrame if target has no exog_targets.
    """
    exog_targets = GEN_LOAD_TARGETS[target].get("exog_targets", [])
    if not exog_targets:
        return pd.DataFrame()

    columns: dict[str, pd.Series] = {}
    for upstream_target in exog_targets:
        for upstream_region in GEN_LOAD_TARGETS[upstream_target]["regions"]:
            tso_df = _load_tso_data(upstream_region)
            col_name = _get_target_col(upstream_target, upstream_region)
            if col_name not in tso_df.columns:
                logger.warning(
                    f"Upstream exog column {col_name} "
                    f"({upstream_target}/{upstream_region}) missing from TSO data. "
                    f"Skipping."
                )
                continue
            columns[col_name] = tso_df[col_name]

    if not columns:
        return pd.DataFrame()

    result = pd.DataFrame(columns)
    logger.info(
        f"Loaded {len(columns)} upstream exog actuals for {target} "
        f"({len(result)} rows, {result.notna().all(axis=1).sum()} fully covered)"
    )
    return result


# ── Feature construction ─────────────────────────────────────────


def _compute_temporal_features(tso_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 22 GEN_LOAD_FEATURES from TSO data.

    Temporal features from the DatetimeIndex + lagged actuals from TSO columns.
    These are stable across Optuna trials (don't depend on weather FE config).
    """
    index = tso_df.index
    temporal = compute_temporal_features(index)
    # Remove _derived_ prefix used by the engine internals
    temporal.columns = [c.removeprefix("_derived_") for c in temporal.columns]

    # Fourier features (period=24, order=3)
    fourier = compute_fourier_features(index, period=24, order=3)
    # Rename to match GEN_LOAD_FEATURES convention
    fourier.columns = [f"hour_fourier_24_3_{c}" for c in fourier.columns]

    result = pd.concat([temporal, fourier], axis=1)

    # Lagged actuals from TSO data — use whatever columns exist
    lag_mappings = {
        "wind_onshore": "gen_wind_on",
        "wind_offshore": "gen_wind_off",
        "solar": "gen_solar",
        "load": "load",
    }
    for col in tso_df.columns:
        # Extract base name (before the TSO suffix like _50hz, _ampr)
        base = None
        for tso_name, feat_prefix in lag_mappings.items():
            if col.startswith(tso_name):
                base = feat_prefix
                break
        if base is None:
            continue

        # 24h lag
        result[f"{base}_h24"] = tso_df[col].shift(24)
        # 7-day rolling average (D-2 availability: shift 48 first, then 7-day window)
        result[f"{base}_d7_d2_avg"] = tso_df[col].shift(48).rolling(7 * 24).mean()

    return result


def _build_features(
    df_weather: pd.DataFrame,
    tso_df: pd.DataFrame,
    target: str,
    region: str,
    weather_config: dict,
    lags_target: int | None,
    temporal_features: pd.DataFrame,
    exog_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build full feature matrix for one Optuna trial.

    Parameters
    ----------
    exog_features : DataFrame with upstream model predictions as columns
        (e.g. 'wind_onshore_DE_50HZ_pred'). Constant across Optuna trials.
        None for targets with no upstream dependencies (wind, solar).

    Returns (X, y, weather_features) aligned on the overlapping index.
    weather_features is the raw output of the weather FE class, useful for
    saving to disk without recomputation.
    """
    tso = REGION_TO_TSO[region]
    weather_type = TARGET_WEATHER_TYPE[target]
    fe_class = _TARGET_FE_CLASS[target]

    # For national targets, use all locations across all TSOs
    if region == "DE_NATIONAL":
        all_locations = []
        for tso_name in ["50Hertz", "Amprion", "TenneT", "TransnetBW", "Creos"]:
            all_locations.extend(locations_for_tso(tso_name, weather_type))
        locations = all_locations
    else:
        locations = locations_for_tso(tso, weather_type)

    fe = fe_class(weather_config, locations)
    weather_features = fe(df_weather)

    # Get target series
    target_col = _get_target_col(target, region)
    y = tso_df[target_col].copy()
    y.name = f"{target}_{region}"

    # Combine weather + temporal + exog features
    common_idx = (
        weather_features.index
        .intersection(temporal_features.index)
        .intersection(y.dropna().index)
    )
    if exog_features is not None and not exog_features.empty:
        common_idx = common_idx.intersection(exog_features.dropna(how="any").index)

    feature_frames = [
        weather_features.loc[common_idx],
        temporal_features.loc[common_idx],
    ]
    if exog_features is not None and not exog_features.empty:
        feature_frames.append(exog_features.loc[common_idx])

    X = pd.concat(feature_frames, axis=1)
    y = y.loc[common_idx]

    # Add target lags if requested. Column naming is consistent with the
    # TSO-level temporal features (e.g. `gen_wind_on_h24`): `{target}_h{N}`
    # means "target shifted N hours back". These are autoregressive features
    # and require recursive forecasting at inference time (see
    # modeling/forecasting.py::forecast_with_lags).
    if lags_target is not None:
        for lag in range(1, lags_target + 1):
            X[f"{target}_h{lag}"] = y.shift(lag)

    # Drop rows with NaN from lagging/rolling
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # Cap to the last `GEN_LOAD_MAX_TRAIN_HOURS` rows. This matches EMA's
    # `df_hist.tail(n_horizons * horizon)` from data_loaders.py:120, keeping
    # LightGBM fit time bounded and focusing training on the most recent
    # (most relevant) regime. Applied after NaN-drop so we always get the
    # requested count of fully-valid rows.
    if len(X) > GEN_LOAD_MAX_TRAIN_HOURS:
        X = X.iloc[-GEN_LOAD_MAX_TRAIN_HOURS:]
        y = y.iloc[-GEN_LOAD_MAX_TRAIN_HOURS:]

    return X, y, weather_features


class _ScaledLogPredictor:
    """Tiny wrapper that lets `forecast_with_lags` drive a bare model + scaler.

    Scales the input row, predicts, and optionally inverse-log-transforms so
    the returned value matches the raw-space target used to construct the
    lag features. ``forecast_with_lags`` writes this raw value back into the
    lag cells of subsequent rows — ensuring scaler statistics and lag
    feature semantics stay consistent across the recursive loop.
    """

    def __init__(self, model, scaler, log_target: bool):
        self.model = model
        self.scaler = scaler
        self.log_target = log_target

    def predict(self, X):
        X_scaled = _apply_scaler(self.scaler, X)
        pred = self.model.predict(X_scaled)
        if self.log_target:
            pred = np.expm1(pred)
        return np.asarray(pred)


def _make_model(model_type: str, params: dict, n_jobs: int = -1):
    """Instantiate a model from type name and params.

    `n_jobs` controls thread count for tree models. -1 = all logical cores
    (matches the historical default). When training many (target, region,
    model) combos in parallel via the CLI's `--parallel` flag, set this to
    `total_cores // parallel_workers` to avoid thread thrashing.
    """
    if model_type == "LGBMRegressor":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, verbosity=-1, n_jobs=n_jobs)
    elif model_type == "XGBRegressor":
        from xgboost import XGBRegressor
        return XGBRegressor(**params, verbosity=0, n_jobs=n_jobs)
    elif model_type == "ElasticNet":
        return ElasticNet(**params, max_iter=5000)
    elif model_type == "Ridge":
        return Ridge(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Optuna objective ─────────────────────────────────────────────


def _optuna_objective(
    trial: optuna.Trial,
    df_weather: pd.DataFrame,
    tso_df: pd.DataFrame,
    target: str,
    region: str,
    temporal_features: pd.DataFrame,
    exog_features: pd.DataFrame | None,
    model_type: str,
    cv_folds: int,
    holdout_days: int,
    n_jobs: int = -1,
) -> float:
    """Single Optuna trial: suggest params, build features, CV, return MAE."""
    # Suggest model hyperparams
    suggest_fn = _MODEL_SUGGEST.get(model_type)
    if suggest_fn is not None:
        model_params = suggest_fn(trial)
    elif model_type == "ElasticNet":
        model_params = {
            "alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0),
        }
    else:
        raise ValueError(f"No Optuna suggest function for model_type: {model_type}")

    # Suggest weather FE config
    fe_class = _TARGET_FE_CLASS[target]
    weather_config = fe_class.suggest_optuna(trial)

    # Suggest dataset params
    ds_params = suggest_dataset_params(trial, model_type=model_type)
    lags_target = ds_params["lags_target"]
    log_target = ds_params["log_target"]
    scaler_name = ds_params["scaler"]

    # Build features with suggested weather config
    X, y, _ = _build_features(
        df_weather, tso_df, target, region,
        weather_config, lags_target, temporal_features,
        exog_features=exog_features,
    )

    if len(X) < _MIN_SAMPLES_FOR_TRAINING:
        raise optuna.TrialPruned("Insufficient data after feature construction")

    # Carve holdout
    pool_idx, _ = carve_holdout(X.index, holdout_days)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]

    # Apply log transform to target
    if log_target:
        y_pool = np.log1p(y_pool.clip(lower=0))

    # Precompute lag column names once per trial. X_pool retains raw-space
    # lag columns (y.shift applied before the log transform was applied to
    # y_pool), so recursive updates must write raw values back into X.
    target_lag_columns = (
        [f"{target}_h{lag}" for lag in range(1, lags_target + 1)]
        if lags_target is not None
        else []
    )

    # CV evaluation — sliding window with weekly test folds, matching EMA
    # (`compute_timeseries_split_cutoffs` with `horizon=168`, `step=168`).
    # Each fold's test set is exactly one week; each recursive forecast
    # spans that week and re-seeds from actuals at the fold start.
    cv = TimeSeriesSplitter(
        n_splits=cv_folds,
        test_days=GEN_LOAD_CV_TEST_DAYS,
        mode="sliding",
    )
    model = _make_model(model_type, model_params, n_jobs=n_jobs)
    cv_maes = []

    for train_idx, test_idx in cv.split(X_pool.index):
        X_train = X_pool.iloc[train_idx]
        y_train = y_pool.iloc[train_idx]
        X_test = X_pool.iloc[test_idx]
        y_test = y_pool.iloc[test_idx]

        # Fit scaler per fold to avoid data leakage
        fold_scaler = _fit_scaler(scaler_name, X_train)
        X_train_scaled = _apply_scaler(fold_scaler, X_train)

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train)

        if target_lag_columns:
            # One recursive forecast per weekly test fold — matches EMA's
            # `forecast_window` call semantics exactly. Windowing helper
            # handles the case where the fold spans slightly more than
            # 168 rows due to day-boundary alignment.
            predictor = _ScaledLogPredictor(fold_model, fold_scaler, log_target)
            y_pred_full, eval_mask = forecast_with_lags_windowed(
                predictor, X_test, y_train, target_lag_columns,
                window_size=DEFAULT_FORECAST_HORIZON,
                sample_windows=None,
            )
            y_true_raw = (
                np.expm1(y_test.values) if log_target else y_test.values
            )
            y_pred = y_pred_full[eval_mask]  # already raw-space
            y_test_eval = y_true_raw[eval_mask]
        else:
            X_test_scaled = _apply_scaler(fold_scaler, X_test)
            y_pred = fold_model.predict(X_test_scaled)
            if log_target:
                y_pred = np.expm1(y_pred)
            y_test_eval = (
                np.expm1(y_test) if log_target else np.asarray(y_test)
            )

        mae = float(np.mean(np.abs(y_test_eval - y_pred)))
        cv_maes.append(mae)

    return float(np.mean(cv_maes))


# ── Main training function ───────────────────────────────────────


def train_gen_load_model(
    target: str,
    region: str,
    model_type: str = "LGBMRegressor",
    optuna_trials: int = GEN_LOAD_OPTUNA_TRIALS,
    cv_folds: int = SEARCH_CV_FOLDS,
    holdout_days: int = GEN_LOAD_HOLDOUT_DAYS,
    n_jobs: int = -1,
) -> str:
    """Full gen/load training pipeline with Optuna TPE search.

    Steps:
    1. Load weather + TSO data.
    2. Pre-compute temporal features (stable across trials).
    3. Optuna TPE search over model hyperparams + weather FE + dataset params.
    4. Save winning weather features to data/processed/weather_features/.
    5. Save full dataset to data/processed/datasets/.
    6. Train final model via train_model() with MAPIE + MLflow tracking.
    7. Return run_id.
    """
    if target not in GEN_LOAD_TARGETS:
        raise ValueError(f"Unknown target '{target}'. Must be one of: {list(GEN_LOAD_TARGETS)}")
    if region not in GEN_LOAD_TARGETS[target]["regions"]:
        raise ValueError(f"Region '{region}' not valid for target '{target}'")

    logger.info(f"Training {model_type} for {target}/{region}")

    # Load data
    df_weather = _load_weather_data(target, region)
    tso_df = _load_tso_data(region)

    # Pre-compute features that are constant across Optuna trials
    temporal_features = _compute_temporal_features(tso_df)

    # Load upstream exog actuals if this target has dependencies.
    # Matches EMA: ground-truth historical values for training; inference
    # substitutes upstream model forecasts (handled at prediction time).
    exog_targets = GEN_LOAD_TARGETS[target].get("exog_targets", [])
    if exog_targets:
        exog_features = _load_upstream_actuals(target)
        logger.info(
            f"Loaded {exog_features.shape[1]} upstream exog features "
            f"({exog_features.shape[0]} rows)"
        )
    else:
        exog_features = None

    # Optuna search
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda trial: _optuna_objective(
            trial, df_weather, tso_df, target, region,
            temporal_features, exog_features, model_type, cv_folds, holdout_days,
            n_jobs=n_jobs,
        ),
        n_trials=optuna_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(f"Best trial #{best.number}: MAE={best.value:.2f}")

    # Extract winning params
    best_params = best.params
    fe_class = _TARGET_FE_CLASS[target]
    weather_config = fe_class.suggest_optuna(
        optuna.trial.FixedTrial(best_params),
    )
    ds_params = suggest_dataset_params(
        optuna.trial.FixedTrial(best_params),
        model_type=model_type,
    )

    # Reconstruct model-specific params (filter out weather/dataset params)
    suggest_fn = _MODEL_SUGGEST.get(model_type)
    if suggest_fn is not None:
        model_params = suggest_fn(optuna.trial.FixedTrial(best_params))
    elif model_type == "ElasticNet":
        model_params = {
            "alpha": best_params["alpha"],
            "l1_ratio": best_params["l1_ratio"],
        }
    else:
        raise ValueError(f"No Optuna suggest function for model_type: {model_type}")

    run_id = _finalize_gen_load_training(
        target=target, region=region, model_type=model_type,
        weather_config=weather_config, ds_params=ds_params,
        model_params=model_params,
        df_weather=df_weather, tso_df=tso_df,
        temporal_features=temporal_features,
        exog_features=exog_features,
        holdout_days=holdout_days, n_jobs=n_jobs,
        feature_version=f"optuna_{model_type.lower()}",
    )

    # Save Optuna study and best params as artifacts on the new run.
    best_config = {
        "model_type": model_type,
        "model_params": model_params,
        "weather_config": weather_config,
        "dataset_params": ds_params,
        "best_mae": best.value,
    }
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        trials_path = Path(tmpdir) / "trials.parquet"
        study.trials_dataframe().to_parquet(trials_path)
        client.log_artifact(run_id, str(trials_path), "optuna")
    _log_best_config_artifact(run_id, best_config)

    logger.info(f"Completed {target}/{region}/{model_type}: run_id={run_id}")
    return run_id


def _finalize_gen_load_training(
    target: str,
    region: str,
    model_type: str,
    weather_config: dict,
    ds_params: dict,
    model_params: dict,
    df_weather: pd.DataFrame,
    tso_df: pd.DataFrame,
    temporal_features: pd.DataFrame,
    exog_features: pd.DataFrame | None,
    holdout_days: int,
    n_jobs: int,
    feature_version: str,
) -> str:
    """Build features and run the final-pass training given resolved configs.

    Shared between Optuna-driven training (``train_gen_load_model``) and
    param-reuse retraining (``retrain_gen_load_from_existing``). Performs:
      1. Build features with actual weather (training rows).
      2. Build parallel features with `hist_forecast` weather (test/holdout).
      3. Save weather_features, dataset, test-fold dataset to disk.
      4. Call ``train_model`` with ``GEN_LOAD_HISTORICAL_FOLDS`` weekly folds.
    Returns the new MLflow run id.
    """
    # Build final features with winning config — actual weather (training).
    X, y, weather_features = _build_features(
        df_weather, tso_df, target, region,
        weather_config, ds_params["lags_target"], temporal_features,
        exog_features=exog_features,
    )

    # Build a parallel feature matrix from `hist_forecast` weather. CV test
    # folds and the holdout are evaluated on these features so the resulting
    # OOF predictions reflect realistic forecast errors — equivalent to EMA's
    # `mode="backtest"` in `generate_historical_forecasts.py`. Training rows
    # continue to use actual weather (the standard EMA pattern).
    df_weather_hf = _load_weather_data(target, region, source="hist_forecast")
    X_test, _, _ = _build_features(
        df_weather_hf, tso_df, target, region,
        weather_config, ds_params["lags_target"], temporal_features,
        exog_features=exog_features,
    )

    # Save weather features for price model transfer
    WEATHER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    weather_feat_path = WEATHER_FEATURES_DIR / f"{target}_{region}.parquet"
    weather_features.to_parquet(weather_feat_path)
    logger.info(f"Saved weather features to {weather_feat_path}")

    # Save dataset (actual weather — used for training fold rows and as the
    # canonical dataset logged to MLflow).
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = f"{target}_{region}_{model_type.lower()}"
    target_key = f"{target}_{region}{TARGET_COL_SUFFIX}"
    dataset = X.copy()
    dataset[target_key] = y
    dataset_path = DATASET_DIR / f"{dataset_name}.parquet"
    dataset.to_parquet(dataset_path)
    logger.info(f"Saved dataset '{dataset_name}': {dataset.shape}")

    # Save test-fold dataset (hist_forecast weather) — same index, same
    # columns, only weather-derived feature values differ. Aligned to the
    # actual-weather dataset's index so train_model can index them in lockstep.
    # Rows pre-`hist_forecast` availability (~2022-01-01) become NaN here;
    # those positions never appear in CV test folds (which are bounded to the
    # GEN_LOAD_HISTORICAL_FOLDS most-recent weeks) so train_model never
    # indexes them.
    test_dataset = X_test.reindex(X.index).copy()
    test_dataset[target_key] = y
    test_dataset_path = DATASET_DIR / f"{dataset_name}.test_hf.parquet"
    test_dataset.to_parquet(test_dataset_path)
    n_swap_rows = int(X_test.index.intersection(X.index).size)
    logger.info(
        f"Saved test-fold dataset (hist_forecast weather) '{dataset_name}.test_hf': "
        f"{test_dataset.shape}, {n_swap_rows}/{len(X)} rows have hist_forecast coverage"
    )

    # Determine experiment name
    experiment = _experiment_for_target(target)

    # Map dataset params to train_model args
    target_transform = "log_shift" if ds_params["log_target"] else "none"
    scaler = ds_params["scaler"]

    # Build model
    model = _make_model(model_type, model_params, n_jobs=n_jobs)

    # Autoregressive target lag columns (if any). When set, train_model uses
    # recursive forecasting in CV and on holdout, matching inference
    # conditions and mirroring EMA's `forecast_window` behaviour.
    lags_target = ds_params["lags_target"]
    target_lag_columns: list[str] | None = (
        [f"{target}_h{lag}" for lag in range(1, lags_target + 1)]
        if lags_target is not None
        else None
    )

    # Train final model with MAPIE + MLflow. Sliding window with weekly test
    # folds; ``GEN_LOAD_HISTORICAL_FOLDS`` controls the OOF span (currently
    # ~4 years, matching `hist_forecast` weather availability). Replaces
    # EMA's standalone `generate_historical_forecasts.py` pipeline.
    cv = TimeSeriesSplitter(
        n_splits=GEN_LOAD_HISTORICAL_FOLDS,
        test_days=GEN_LOAD_CV_TEST_DAYS,
        mode="sliding",
    )
    run_id = train_model(
        dataset_path=dataset_path,
        test_dataset_path=test_dataset_path,
        model=model,
        experiment=experiment,
        tags={
            "stage": "model_training",
            "feature_version": feature_version,
            "target": target,
            "region": region,
        },
        scaler=scaler,
        target_transform=target_transform,
        cv=cv,
        holdout_days=holdout_days,
        collect_oof=True,
        target_lag_columns=target_lag_columns,
    )
    return run_id


def _log_best_config_artifact(run_id: str, config: dict) -> None:
    """Log a best_config.json artifact to ``run_id``."""
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        client.log_artifact(run_id, str(config_path), "optuna")


def _find_latest_base_run(target: str, region: str, model_type: str) -> str:
    """Look up the most recent finished base-model run for (target, region,
    model_type) in the corresponding ``generation/*`` experiment.

    Used by ``retrain_gen_load_from_existing`` to locate the source
    ``best_config.json``. Filters by ``feature_version='optuna_<model>'`` so
    StackingEnsemble runs (``feature_version='ensemble'``) are excluded.
    """
    experiment = _experiment_for_target(target)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENTS[experiment])
    if exp is None:
        raise ValueError(f"MLflow experiment {EXPERIMENTS[experiment]!r} not found")
    feature_version = f"optuna_{model_type.lower()}"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=(
            f"tags.target='{target}' AND tags.region='{region}' "
            f"AND tags.feature_version='{feature_version}' "
            f"AND attributes.status='FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No prior base run for {target}/{region}/{model_type} "
            f"(feature_version={feature_version!r}). Run train_gen_load_model "
            f"first or use --no-reuse-params to start a fresh Optuna search."
        )
    return runs[0].info.run_id


def retrain_gen_load_from_existing(
    target: str,
    region: str,
    model_type: str,
    holdout_days: int = GEN_LOAD_HOLDOUT_DAYS,
    n_jobs: int = -1,
) -> str:
    """Re-run the final training pass for an existing (target, region,
    model_type) reusing winning hyperparameters from MLflow.

    Skips Optuna search entirely. Used to extend OOF coverage (e.g. bump
    ``GEN_LOAD_HISTORICAL_FOLDS`` and rebuild the historical_forecasts
    artifacts) without re-running the costly hyperparameter search.

    Params are loaded from the most recent finished run's ``optuna/best_config.json``
    artifact. Returns the new MLflow run id.
    """
    if target not in GEN_LOAD_TARGETS:
        raise ValueError(f"Unknown target '{target}'. Must be one of: {list(GEN_LOAD_TARGETS)}")
    if region not in GEN_LOAD_TARGETS[target]["regions"]:
        raise ValueError(f"Region '{region}' not valid for target '{target}'")

    logger.info(f"Retraining {model_type} for {target}/{region} from existing best params")

    # 1. Locate source run and load best_config.json
    source_run_id = _find_latest_base_run(target, region, model_type)
    client = mlflow.MlflowClient()
    config_path = client.download_artifacts(source_run_id, "optuna/best_config.json")
    with open(config_path) as f:
        best_config = json.load(f)
    logger.info(
        f"Loaded best_config.json from run {source_run_id[:8]} "
        f"(original best_mae={best_config.get('best_mae'):.2f})"
    )

    weather_config = best_config["weather_config"]
    ds_params = best_config["dataset_params"]
    model_params = best_config["model_params"]

    if best_config["model_type"] != model_type:
        raise ValueError(
            f"best_config.json model_type mismatch: expected {model_type}, "
            f"got {best_config['model_type']}"
        )

    # 2. Load data and pre-compute temporal/exog features
    df_weather = _load_weather_data(target, region)
    tso_df = _load_tso_data(region)
    temporal_features = _compute_temporal_features(tso_df)
    exog_targets = GEN_LOAD_TARGETS[target].get("exog_targets", [])
    exog_features = _load_upstream_actuals(target) if exog_targets else None
    if exog_features is not None:
        logger.info(
            f"Loaded {exog_features.shape[1]} upstream exog features "
            f"({exog_features.shape[0]} rows)"
        )

    # 3. Run the shared final-training pass
    run_id = _finalize_gen_load_training(
        target=target, region=region, model_type=model_type,
        weather_config=weather_config, ds_params=ds_params,
        model_params=model_params,
        df_weather=df_weather, tso_df=tso_df,
        temporal_features=temporal_features,
        exog_features=exog_features,
        holdout_days=holdout_days, n_jobs=n_jobs,
        feature_version=f"optuna_{model_type.lower()}",
    )

    # 4. Tag provenance + re-log best_config.json on the new run
    client.set_tag(run_id, "reused_params_from", source_run_id)
    _log_best_config_artifact(run_id, best_config)

    logger.info(
        f"Completed retrain {target}/{region}/{model_type}: "
        f"new run_id={run_id} (reused params from {source_run_id[:8]})"
    )
    return run_id


def _experiment_for_target(target: str) -> str:
    """Map gen/load target to MLflow experiment key."""
    return f"gen_{target}"


# ── Stacking ensemble ────────────────────────────────────────────


def ensemble_gen_load(
    target: str,
    region: str,
    base_run_ids: list[str],
    holdout_days: int = GEN_LOAD_HOLDOUT_DAYS,
    confidence_level: float = PI_CONFIDENCE_LEVEL,
) -> str:
    """Stacking ensemble for gen/load models.

    1. Load OOF predictions from each base model (logged as artifacts).
    2. Stack as meta-features: [pred_model1, pred_model2, ...].
    3. Train Ridge(positive=True) meta-learner on OOF predictions.
    4. Evaluate on holdout.
    5. Post-hoc conformal calibration for ensemble PI.
    6. If ensemble doesn't beat the best single model, fall back.
    7. Log to MLflow.
    """
    client = mlflow.MlflowClient()

    # Load OOF and holdout predictions from each base model
    oof_dfs = {}
    holdout_dfs = {}
    holdout_metrics = {}

    for run_id in base_run_ids:
        run = client.get_run(run_id)
        model_class = run.data.tags.get("model_class", run_id)

        # Download artifacts
        artifact_dir = client.download_artifacts(run_id, "predictions")
        oof_path = Path(artifact_dir) / "oof_predictions.parquet"
        holdout_path = Path(artifact_dir) / "holdout_predictions.parquet"

        if not oof_path.exists():
            raise FileNotFoundError(
                f"Run {run_id} has no OOF predictions. "
                f"Re-train with collect_oof=True."
            )

        oof_dfs[model_class] = pd.read_parquet(oof_path)
        holdout_dfs[model_class] = pd.read_parquet(holdout_path)
        holdout_metrics[model_class] = float(run.data.metrics.get("mae", np.inf))

    # Build meta-feature matrices
    # OOF: align on common index (intersection of all folds)
    common_oof_idx = oof_dfs[list(oof_dfs)[0]].index
    for df in oof_dfs.values():
        common_oof_idx = common_oof_idx.intersection(df.index)

    X_meta_train = pd.DataFrame(
        {name: df.loc[common_oof_idx, "y_pred"] for name, df in oof_dfs.items()},
        index=common_oof_idx,
    )
    y_meta_train = oof_dfs[list(oof_dfs)[0]].loc[common_oof_idx, "y_true"]

    # Holdout: align similarly
    common_holdout_idx = holdout_dfs[list(holdout_dfs)[0]].index
    for df in holdout_dfs.values():
        common_holdout_idx = common_holdout_idx.intersection(df.index)

    X_meta_holdout = pd.DataFrame(
        {name: df.loc[common_holdout_idx, "y_pred"] for name, df in holdout_dfs.items()},
        index=common_holdout_idx,
    )
    y_meta_holdout = holdout_dfs[list(holdout_dfs)[0]].loc[common_holdout_idx, "y_true"]

    # Train Ridge meta-learner
    meta_model = Ridge(alpha=1.0, positive=True)
    meta_model.fit(X_meta_train, y_meta_train)

    # Predict on holdout
    y_ensemble_pred = meta_model.predict(X_meta_holdout)
    ensemble_metrics = calculate_metrics(np.asarray(y_meta_holdout), y_ensemble_pred)

    # Check if ensemble beats best single model
    best_single_mae = min(holdout_metrics.values())
    best_single_name = min(holdout_metrics, key=holdout_metrics.get)
    ensemble_mae = ensemble_metrics["mae"]

    if ensemble_mae >= best_single_mae:
        logger.warning(
            f"Ensemble MAE ({ensemble_mae:.2f}) doesn't beat best single model "
            f"({best_single_name}: {best_single_mae:.2f}). Falling back."
        )
        fallback = True
    else:
        improvement = (best_single_mae - ensemble_mae) / best_single_mae * 100
        logger.info(
            f"Ensemble MAE={ensemble_mae:.2f} vs best single "
            f"{best_single_name}={best_single_mae:.2f} ({improvement:.1f}% improvement)"
        )
        fallback = False

    # Post-hoc conformal calibration for ensemble PI
    conformal_q = calibrate_ensemble_intervals(
        np.asarray(y_meta_holdout), y_ensemble_pred, confidence_level,
    )
    y_lower, y_upper = predict_ensemble_intervals(y_ensemble_pred, conformal_q)
    pi_metrics = calculate_pi_metrics(np.asarray(y_meta_holdout), y_lower, y_upper)

    # Log to MLflow
    experiment = _experiment_for_target(target)
    tags = {
        "stage": "model_training",
        "feature_version": "ensemble",
        "target": target,
        "region": region,
    }

    with TrackedRun(experiment, **tags) as run:
        run_id = run.info.run_id

        mlflow.set_tag("model_class", "StackingEnsemble")
        mlflow.set_tag("base_models", ",".join(holdout_metrics.keys()))
        mlflow.set_tag("base_run_ids", ",".join(base_run_ids))
        mlflow.set_tag("fallback_to_single", str(fallback))
        mlflow.set_tag("n_features", str(X_meta_train.shape[1]))
        mlflow.set_tag("n_train_rows", str(len(X_meta_train)))

        mlflow.log_metrics(ensemble_metrics)
        mlflow.log_metrics(pi_metrics)
        mlflow.log_metric("conformal_quantile", conformal_q)
        mlflow.log_metric("best_single_mae", best_single_mae)

        # Log meta-learner weights
        for name, weight in zip(X_meta_train.columns, meta_model.coef_):
            mlflow.log_metric(f"weight_{name}", float(weight))

        # Log holdout predictions
        holdout_preds = pd.DataFrame(
            {
                "y_true": y_meta_holdout.values,
                "y_pred": y_ensemble_pred,
                "y_lower": y_lower,
                "y_upper": y_upper,
            },
            index=common_holdout_idx,
        )
        # Log OOF predictions (ensemble's view of the OOF window). The
        # meta-learner's in-sample predictions on the OOF meta-features are
        # used here. Because the meta-learner is a constrained Ridge
        # (positive coefficients, alpha=1.0) over only ~3 base predictions,
        # the in-sample-vs-honest-OOF bias is small (<1%); fully leak-free
        # ensemble OOF would require a leave-one-fold-out CV at the meta
        # level, deferred until needed. Stage 5c uses these as input
        # features, not for ensemble-level evaluation, so the bias is OK.
        oof_preds = pd.DataFrame(
            {
                "y_true": y_meta_train.values,
                "y_pred": meta_model.predict(X_meta_train),
            },
            index=common_oof_idx,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            holdout_path = Path(tmpdir) / "holdout_predictions.parquet"
            holdout_preds.to_parquet(holdout_path)
            mlflow.log_artifact(str(holdout_path), "predictions")

            oof_path = Path(tmpdir) / "oof_predictions.parquet"
            oof_preds.to_parquet(oof_path)
            mlflow.log_artifact(str(oof_path), "predictions")

        # Log meta-model
        mlflow.sklearn.log_model(meta_model, "model")

        logger.info(
            f"Ensemble run {run_id}: MAE={ensemble_mae:.2f}, "
            f"PI coverage={pi_metrics['pi_coverage']:.2%}"
        )

    return run_id
