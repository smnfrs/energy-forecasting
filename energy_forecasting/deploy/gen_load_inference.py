"""Daily gen/load inference pipeline.

Produces 168h (1-week) forecasts for each (target, region) combination using
models from models/gen_load_config.json. Inference respects the same training
order (wave 1: wind/solar → wave 2: load → wave 3: gen_load_diff).

Feature reconstruction mirrors training exactly:
- Weather FE class re-instantiated with saved weather_config
- Temporal features computed from forecast timestamps
- TSO lag features (h24, d7_d2_avg) forward-filled from last known actuals
- Exog features from previous wave outputs (not SMARD actuals as at training time)
- Target lag columns pre-seeded from last actuals, updated recursively by forecast
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config import (
    HISTORICAL_FORECASTS_DIR,
    RAW_DATA_DIR,
)
from energy_forecasting.config.locations import locations_for_tso
from energy_forecasting.config.modeling import (
    GEN_LOAD_TARGETS,
    GEN_LOAD_TRAINING_ORDER,
    REGION_TO_TSO,
    TARGET_WEATHER_TYPE,
)
from energy_forecasting.config.smard import TSO_REGIONS, TSO_SUFFIXES
from energy_forecasting.deploy.model_store import (
    GEN_LOAD_CONFIG_PATH,
    load_gen_load_config,
    load_gen_load_model,
)
from energy_forecasting.features.market import compute_fourier_features, compute_temporal_features
from energy_forecasting.features.weather_load import WeatherLoadFE
from energy_forecasting.features.weather_solar import WeatherSolarPowerFE
from energy_forecasting.features.weather_wind import WeatherWindPowerFE
from energy_forecasting.modeling.forecasting import (
    DEFAULT_FORECAST_HORIZON,
    find_target_lag_columns,
    forecast_direct,
    forecast_with_lags,
)

_TARGET_FE_CLASS: dict[str, type] = {
    "wind_onshore": WeatherWindPowerFE,
    "wind_offshore": WeatherWindPowerFE,
    "solar": WeatherSolarPowerFE,
    "load": WeatherLoadFE,
    "gen_load_diff": WeatherLoadFE,
}

# Lookback needed to seed rolling lag features (7-day avg shifted 48h = 216h)
_TEMPORAL_LOOKBACK = 240


def _load_tso_data(region: str) -> pd.DataFrame:
    """Load TSO parquet (or aggregate national) for lag feature seeding."""
    from energy_forecasting.modeling.gen_load import (
        _load_national_tso_data,
    )
    from energy_forecasting.modeling.gen_load import (
        _load_tso_data as _training_load_tso,
    )

    if region == "DE_NATIONAL":
        return _load_national_tso_data()
    return _training_load_tso(region)


def _get_target_col(target: str, region: str) -> str:
    """Target column name inside the TSO parquet."""
    from energy_forecasting.modeling.gen_load import _get_target_col as _training_get

    return _training_get(target, region)


def _load_forecast_weather(target: str, region: str) -> pd.DataFrame:
    """Load current-forecast weather (Open-Meteo 14-day ahead)."""
    weather_type = TARGET_WEATHER_TYPE[target]
    if region == "DE_NATIONAL":
        frames = []
        for tso_name in TSO_REGIONS:
            path = RAW_DATA_DIR / "weather" / "cities" / tso_name / "forecast.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            raise FileNotFoundError("No national forecast weather parquets found")
        return pd.concat(frames, axis=1)
    tso_name = REGION_TO_TSO[region]
    path = RAW_DATA_DIR / "weather" / weather_type / tso_name / "forecast.parquet"
    return pd.read_parquet(path)


def _get_locations(target: str, region: str) -> list:
    """Location list for weather FE class, matching training."""

    weather_type = TARGET_WEATHER_TYPE[target]
    if region == "DE_NATIONAL":
        locs = []
        for tso_name in TSO_REGIONS:
            locs.extend(locations_for_tso(tso_name, "cities"))
        return locs
    tso_name = REGION_TO_TSO[region]
    return locations_for_tso(tso_name, weather_type)


def _build_temporal_and_lag_features(
    tso_df: pd.DataFrame,
    forecast_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute temporal + TSO lag features for the forecast window.

    Extends the TSO data with forward-filled actuals for the forecast period so
    that shift() and rolling() operations produce deterministic values. For lag
    features beyond the lookback window (h>24 for h24 lag, h>48 for d7_d2_avg),
    this is an approximation (last-known-actual forward fill). The target lag
    columns ({target}_h{N}) are handled separately and NOT computed here.
    """
    lookback = _TEMPORAL_LOOKBACK
    tso_recent = tso_df.iloc[-lookback:].copy()

    # Extend with forward-filled NaN rows for the forecast period
    empty = pd.DataFrame(
        np.nan,
        index=forecast_idx,
        columns=tso_recent.columns,
    )
    tso_ext = pd.concat([tso_recent, empty])
    tso_ext = tso_ext[~tso_ext.index.duplicated(keep="last")]
    tso_ext = tso_ext.ffill()

    # Temporal + Fourier features (pure datetime — always valid)
    temporal = compute_temporal_features(tso_ext.index)
    temporal.columns = [c.removeprefix("_derived_") for c in temporal.columns]
    fourier = compute_fourier_features(tso_ext.index, period=24, order=3)
    fourier.columns = [f"hour_fourier_24_3_{c}" for c in fourier.columns]
    result = pd.concat([temporal, fourier], axis=1)

    # TSO lag features (h24 and d7_d2_avg), same naming as _compute_temporal_features
    lag_mappings = {
        "wind_onshore": "gen_wind_on",
        "wind_offshore": "gen_wind_off",
        "solar": "gen_solar",
        "load": "load",
    }
    for col in tso_ext.columns:
        base = None
        for tso_name, feat_prefix in lag_mappings.items():
            if col.startswith(tso_name):
                base = feat_prefix
                break
        if base is None:
            continue
        result[f"{base}_h24"] = tso_ext[col].shift(24)
        result[f"{base}_d7_d2_avg"] = tso_ext[col].shift(48).rolling(7 * 24).mean()

    return result.loc[forecast_idx]


def _build_target_lag_seed(
    target: str,
    region: str,
    y_actual: pd.Series,
    lags_target: int,
    forecast_idx: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """Pre-fill autoregressive target lag columns from the last known actuals.

    Returns a dict {col_name: array_of_length_168}. Values beyond the last
    actual are filled with the last known value; forecast_with_lags will
    overwrite them as predictions become available.
    """
    last_val = float(y_actual.dropna().iloc[-1])
    cols: dict[str, np.ndarray] = {}
    for lag in range(1, lags_target + 1):
        col = f"{target}_h{lag}"
        source_idx = forecast_idx - pd.Timedelta(hours=lag)
        values = y_actual.reindex(source_idx).values.astype(float)
        # Forward-fill with last actual where reindex gives NaN
        mask = np.isnan(values)
        values[mask] = last_val
        cols[col] = values
    return cols


def _solar_elevation_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Return True where the sun is above the horizon for Germany.

    Uses approximate solar position for Germany's centre (51.2°N, 10.4°E).
    Accounts for seasonal DST variation — index must be UTC-aware or UTC-naive
    in the sense that hours are UTC hours.
    """
    lat, lon = 51.2, 10.4
    utc_hours = index.hour + index.minute / 60.0
    doy = index.day_of_year
    decl = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    hour_angle = np.radians(15.0 * (utc_hours + lon / 15.0 - 12.0))
    elev = np.arcsin(
        np.sin(np.radians(lat)) * np.sin(decl)
        + np.cos(np.radians(lat)) * np.cos(decl) * np.cos(hour_angle)
    )
    return np.asarray(elev > 0)


def _build_exog_features(
    target: str,
    exog_forecasts: dict[tuple[str, str], pd.DataFrame] | None,
    forecast_idx: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    """Build exog feature DataFrame from upstream wave outputs.

    Uses the same column names as _load_upstream_actuals (e.g. wind_onshore_50hz)
    so _build_features-compatible feature matrices can be constructed.
    """
    if exog_forecasts is None:
        return None
    exog_targets = GEN_LOAD_TARGETS[target].get("exog_targets", [])
    if not exog_targets:
        return None

    cols: dict[str, pd.Series] = {}
    for upstream_target in exog_targets:
        for upstream_region in GEN_LOAD_TARGETS[upstream_target]["regions"]:
            key = (upstream_target, upstream_region)
            if key not in exog_forecasts:
                logger.warning(
                    f"Missing exog forecast for {upstream_target}/{upstream_region} "
                    f"(needed by {target}). Skipping this exog column."
                )
                continue
            upstream_df = exog_forecasts[key]
            if upstream_region == "DE_NATIONAL":
                col_name = _get_target_col(upstream_target, upstream_region)
            else:
                tso_name = REGION_TO_TSO[upstream_region]
                suffix = TSO_SUFFIXES[tso_name]
                col_name = f"{upstream_target}{suffix}"
            # ffill: upstream wave may end 1h early if its TSO data lagged behind;
            # carrying the last value forward avoids a trailing NaN that would
            # otherwise cause dropna() to strip the final forecast hour.
            cols[col_name] = upstream_df["y_pred"].reindex(forecast_idx).ffill()

    if not cols:
        return None
    return pd.DataFrame(cols, index=forecast_idx)


def _infer_one(
    target: str,
    region: str,
    config_entry: dict,
    exog_forecasts: dict[tuple[str, str], pd.DataFrame] | None,
) -> pd.DataFrame:
    """Run 168h inference for a single (target, region) combo.

    Returns DataFrame with columns [y_pred, y_lower, y_upper] indexed by
    hourly forecast timestamps.
    """
    ds_params = config_entry["dataset_params"]
    weather_config = config_entry["weather_config"]
    lags_target = ds_params.get("lags_target")

    # Load model
    model = load_gen_load_model(target, region)

    # Load forecast weather and determine forecast window
    df_weather_fc = _load_forecast_weather(target, region)

    # Load TSO actuals for lag seeding
    tso_df = _load_tso_data(region)
    target_col = _get_target_col(target, region)
    if target_col not in tso_df.columns and target != "gen_load_diff":
        raise KeyError(f"Target column '{target_col}' missing from TSO data for {region}")
    y_actual = tso_df[target_col] if target_col in tso_df.columns else tso_df.iloc[:, 0]

    # Forecast timestamps: from last TSO actual + 1h, for 168h
    last_ts = tso_df.index[-1]
    tz = last_ts.tzinfo
    forecast_idx = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=DEFAULT_FORECAST_HORIZON,
        freq="h",
        tz=tz,
    )

    # Weather FE
    fe_class = _TARGET_FE_CLASS[target]
    locations = _get_locations(target, region)
    fe = fe_class(weather_config, locations)
    weather_features = fe(df_weather_fc)

    # Temporal + TSO lag features
    temporal_features = _build_temporal_and_lag_features(tso_df, forecast_idx)

    # Exog features from previous wave
    exog_df = _build_exog_features(target, exog_forecasts, forecast_idx)

    # Align all feature frames on forecast_idx
    common_idx = forecast_idx.intersection(weather_features.index).intersection(
        temporal_features.dropna(how="any").index
    )
    if exog_df is not None and not exog_df.empty:
        common_idx = common_idx.intersection(exog_df.dropna(how="any").index)

    if len(common_idx) < DEFAULT_FORECAST_HORIZON:
        logger.warning(
            f"{target}/{region}: only {len(common_idx)}/{DEFAULT_FORECAST_HORIZON} "
            "forecast hours have complete features"
        )

    frames = [weather_features.loc[common_idx], temporal_features.loc[common_idx]]
    if exog_df is not None and not exog_df.empty:
        frames.append(exog_df.loc[common_idx])
    X_forecast = pd.concat(frames, axis=1)

    # Target lag columns (autoregressive)
    if lags_target:
        lag_seed = _build_target_lag_seed(target, region, y_actual, lags_target, common_idx)
        for col, vals in lag_seed.items():
            X_forecast[col] = vals

        lag_columns = find_target_lag_columns(X_forecast.columns, target)
        result_df = forecast_with_lags(model, X_forecast, y_actual, lag_columns)
    else:
        result_df = forecast_direct(model, X_forecast)

    # Rename to standard output columns
    out = result_df.rename(columns={"fitted": "y_pred", "lower": "y_lower", "upper": "y_upper"})

    # Apply conformal PI from ensemble run metrics if base model has no intervals
    # (recursive-lag models bypass MAPIE and have NaN in lower/upper)
    if out["y_lower"].isna().all():
        run = _try_get_conformal_quantile(config_entry["run_id"])
        if run is not None:
            out["y_lower"] = out["y_pred"] - run
            out["y_upper"] = out["y_pred"] + run

    # Solar cannot produce at night — zero out predictions below the horizon.
    # Recursive lag forecasts accumulate errors and can predict non-zero values
    # at night; this clamp applies the physical constraint.
    if target == "solar":
        is_day = _solar_elevation_mask(out.index)
        out.loc[~is_day, ["y_pred", "y_lower", "y_upper"]] = 0.0

    logger.info(f"Inferred {target}/{region}: {len(out)} hours, mean={out['y_pred'].mean():.1f}")
    return out


def _try_get_conformal_quantile(run_id: str) -> float | None:
    """Load conformal_quantile metric from an MLflow run, if available."""
    try:
        import mlflow

        from energy_forecasting.config import MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        metrics = mlflow.MlflowClient().get_run(run_id).data.metrics
        q = metrics.get("conformal_quantile")
        return float(q) if q is not None else None
    except Exception:
        return None


def _run_target_wave(
    targets: list[str],
    gl_config: dict,
    exog_forecasts: dict[tuple[str, str], pd.DataFrame] | None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Run inference for all (target, region) combos in a wave."""
    results: dict[tuple[str, str], pd.DataFrame] = {}
    for target in targets:
        for region in GEN_LOAD_TARGETS[target]["regions"]:
            combo_key = f"{target}/{region}"
            if combo_key not in gl_config["combos"]:
                logger.warning(f"No config entry for {combo_key}, skipping")
                continue
            try:
                df = _infer_one(target, region, gl_config["combos"][combo_key], exog_forecasts)
                results[(target, region)] = df
            except Exception:
                logger.exception(f"FAILED: inference for {combo_key}")
    return results


def aggregate_national(
    results: dict[tuple[str, str], pd.DataFrame],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Sum per-TSO forecasts to DE_NATIONAL level.

    Matches the training-time _aggregate_national_historical_forecasts pattern:
    sum y_pred, y_lower, y_upper across all TSO regions for each target.
    The resulting DataFrame is added to results under ('target', 'DE_NATIONAL').
    """
    national: dict[tuple[str, str], pd.DataFrame] = {}
    for target in ["wind_onshore", "wind_offshore", "solar", "load"]:
        regions = GEN_LOAD_TARGETS[target]["regions"]
        frames = [results[(target, r)] for r in regions if (target, r) in results]
        if not frames:
            continue
        # Union so TSOs with slightly different SMARD data extents don't truncate output
        all_idx = frames[0].index
        for df in frames[1:]:
            all_idx = all_idx.union(df.index)
        agg = sum(df.reindex(all_idx, fill_value=0) for df in frames)
        # Cap to expected forecast window (TSO with more-current data may add 1 extra hour)
        if len(agg) > DEFAULT_FORECAST_HORIZON:
            agg = agg.iloc[:DEFAULT_FORECAST_HORIZON]
        national[(target, "DE_NATIONAL")] = agg
        logger.info(
            f"National aggregate {target}: {len(agg)} hours, mean={agg['y_pred'].mean():.1f}"
        )
    return national


def update_historical_forecasts(
    results: dict[tuple[str, str], pd.DataFrame],
) -> None:
    """Append today's live forecasts to the historical_forecasts parquets.

    These files are used by the price model's EMA overlay
    (_overlay_ema_forecasts) to supply the prog_* features for D+1.
    """
    HISTORICAL_FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    for (target, region), df in results.items():
        path = HISTORICAL_FORECASTS_DIR / f"{target}_{region}.parquet"
        new_rows = df.rename(
            columns={"y_pred": "y_pred", "y_lower": "y_lower", "y_upper": "y_upper"}
        )
        # Add placeholder y_true (unknown until next day)
        new_rows = new_rows.copy()
        if "y_true" not in new_rows.columns:
            new_rows["y_true"] = np.nan
        new_rows = new_rows[["y_true", "y_pred", "y_lower", "y_upper"]]

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_rows])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_rows

        combined.to_parquet(path)
        logger.debug(f"Updated historical_forecasts/{target}_{region}.parquet")


def run_gen_load_inference(
    config_path: Path = GEN_LOAD_CONFIG_PATH,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Run 168h gen/load inference for all combos, respecting training wave order.

    Returns dict keyed by (target, region) with DataFrames of
    [y_pred, y_lower, y_upper] indexed by hourly forecast timestamps.
    """
    gl_config = load_gen_load_config()
    logger.info(f"Loaded gen_load_config.json: {len(gl_config['combos'])} combos")

    all_results: dict[tuple[str, str], pd.DataFrame] = {}

    # Wave 1: wind + solar (independent, no exog)
    wave1_targets = GEN_LOAD_TRAINING_ORDER[0]
    logger.info(f"Wave 1: {wave1_targets}")
    wave1 = _run_target_wave(wave1_targets, gl_config, exog_forecasts=None)
    all_results.update(wave1)

    # Wave 2: load (needs wind/solar)
    wave2_targets = GEN_LOAD_TRAINING_ORDER[1]
    logger.info(f"Wave 2: {wave2_targets}")
    wave2 = _run_target_wave(wave2_targets, gl_config, exog_forecasts=all_results)
    all_results.update(wave2)

    # Wave 3: gen_load_diff (needs all above)
    wave3_targets = GEN_LOAD_TRAINING_ORDER[2]
    logger.info(f"Wave 3: {wave3_targets}")
    wave3 = _run_target_wave(wave3_targets, gl_config, exog_forecasts=all_results)
    all_results.update(wave3)

    # National aggregates
    national = aggregate_national(all_results)
    all_results.update(national)

    logger.info(f"Gen/load inference complete: {len(all_results)} targets")
    return all_results
