"""Feature engineering engine — orchestrates parsing, validation, and computation.

Entry points:
- engineer_features(df, feature_list): full feature matrix from merged DataFrame
- extend_features(existing, df, feature_list): compute only new rows, append

The engine parses each feature string, dispatches to the appropriate market
function, and collects results into a single DataFrame.
"""

import pandas as pd
from loguru import logger

from energy_forecasting.features.market import (
    compute_eeg_regime,
    compute_ewma,
    compute_fourier_features,
    compute_generation_pct,
    compute_hourly_lag,
    compute_interaction,
    compute_neg_price_stats,
    compute_net_exports,
    compute_price_spreads,
    compute_rolling_stat,
    compute_temporal_features,
)
from energy_forecasting.features.parser import (
    FeatureSpec,
    InteractionSpec,
    parse_feature,
)
from energy_forecasting.features.validation import validate_features


def _prepare_working_df(df: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    """Pre-compute derived columns needed by any feature in the list.

    Scans the parsed specs for _derived_ column references, batch-computes
    the prerequisite market functions, and returns a working copy of df
    with derived columns added.
    """
    needed_derived = set()
    for spec in specs:
        if spec.raw_col.startswith("_derived_"):
            needed_derived.add(spec.raw_col)

    if not needed_derived:
        return df

    work = df.copy()

    # Temporal features (hour_sin, dow_cos, is_holiday, day_index, etc.)
    temporal_prefixes = (
        "_derived_hour",
        "_derived_dow",
        "_derived_month",
        "_derived_is_weekend",
        "_derived_is_holiday",
        "_derived_day_index",
        "_derived_year_index",
    )
    if any(d.startswith(temporal_prefixes) for d in needed_derived):
        temporal = compute_temporal_features(df.index)
        for col in temporal.columns:
            if col in needed_derived:
                work[col] = temporal[col]
        if "_derived_hour_of_day" in needed_derived:
            work["_derived_hour_of_day"] = df.index.hour.astype(float)
        if "_derived_hour_of_day_int" in needed_derived:
            work["_derived_hour_of_day_int"] = df.index.hour.astype(float)
        if "_derived_day_of_week_int" in needed_derived:
            work["_derived_day_of_week_int"] = df.index.dayofweek.astype(float)

    # Generation percentages (gen_pct_*, total_generation, pct_renewable, supply_demand_gap)
    gen_pct_prefixes = (
        "_derived_gen_pct_",
        "_derived_total_generation",
        "_derived_pct_renewable",
        "_derived_supply_demand_gap",
        "_derived_pct_prog_",
    )
    if any(d.startswith(gen_pct_prefixes) for d in needed_derived):
        gen_pct = compute_generation_pct(
            df,
            add_renewable_pct="_derived_pct_renewable" in needed_derived,
            add_supply_demand_gap="_derived_supply_demand_gap" in needed_derived,
            add_prognosticated_pct=any(d.startswith("_derived_pct_prog_") for d in needed_derived),
        )
        for col in gen_pct.columns:
            if col in needed_derived:
                work[col] = gen_pct[col]

    # Price spreads
    if any(d.startswith("_derived_spread_") for d in needed_derived):
        spreads = compute_price_spreads(df)
        for col in spreads.columns:
            if col in needed_derived:
                work[col] = spreads[col]

    # Net exports (total_exports, total_imports, net_export_*)
    net_export_prefixes = (
        "_derived_net_export_",
        "_derived_total_exports",
        "_derived_total_imports",
    )
    if any(d.startswith(net_export_prefixes) for d in needed_derived):
        net_exports = compute_net_exports(df)
        for col in net_exports.columns:
            if col in needed_derived:
                work[col] = net_exports[col]

    # EEG regime indicator (deterministic from date)
    if "_derived_eeg_regime" in needed_derived:
        work["_derived_eeg_regime"] = compute_eeg_regime(df.index)

    # Negative-price rolling stats on the target price
    neg_price_cols = {d for d in needed_derived if d.startswith("_derived_neg_price_")}
    if neg_price_cols:
        if "target_price" not in df.columns:
            raise KeyError("neg_price_* features require 'target_price' in the input DataFrame.")
        neg_stats = compute_neg_price_stats(df["target_price"])
        for col in neg_stats.columns:
            if col in neg_price_cols:
                work[col] = neg_stats[col]

    return work


def _compute_single(spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
    """Compute a single feature from its parsed spec."""
    raw_col = spec.raw_col

    if raw_col not in df.columns:
        raise KeyError(
            f"Column {raw_col!r} (from short name {spec.base!r}) "
            f"not found in DataFrame. Available: {sorted(df.columns)[:10]}..."
        )

    series = df[raw_col]

    # Apply suffix transformation
    if spec.lag is not None:
        return compute_hourly_lag(series, spec.lag.hours)

    if spec.agg is not None:
        agg = spec.agg
        return compute_rolling_stat(
            series,
            start_day=agg.start_day,
            end_day=agg.end_day,
            stat=agg.stat,
            end_hour=agg.end_hour,
            hour_start=agg.hour_start,
            hour_end=agg.hour_end,
        )

    if spec.ewma is not None:
        return compute_ewma(
            series,
            span=spec.ewma.span,
            cutoff_day=spec.ewma.cutoff_day,
            cutoff_hour=spec.ewma.cutoff_hour,
        )

    if spec.fourier is not None:
        return compute_fourier_features(df.index, spec.fourier.period, spec.fourier.order)

    if spec.daily_agg is not None:
        return compute_rolling_stat(series, start_day=0, end_day=0, stat=spec.daily_agg.stat)

    # No suffix — return as-is (only valid for deterministic/forecast features)
    return series


def engineer_features(
    df: pd.DataFrame,
    feature_list: list[str],
    validate: bool = True,
) -> pd.DataFrame:
    """Build feature matrix from merged DataFrame and feature list.

    1. Validates all features against availability rules (if validate=True)
    2. Parses all feature strings
    3. Pre-computes derived columns (gen_pct, spreads, etc.)
    4. Computes each feature column
    5. Returns DataFrame with feature columns only
    """
    if validate:
        errors = validate_features(feature_list)
        if errors:
            msg = f"{len(errors)} leakage validation error(s):\n"
            for e in errors:
                msg += f"  {e.feature_str}: {e.reason}\n"
            raise ValueError(msg)

    # Parse all specs
    parsed: list[tuple[str, FeatureSpec | InteractionSpec]] = []
    all_simple_specs: list[FeatureSpec] = []
    for feat_str in feature_list:
        spec = parse_feature(feat_str)
        parsed.append((feat_str, spec))
        if isinstance(spec, InteractionSpec):
            all_simple_specs.extend([spec.left, spec.right])
        else:
            all_simple_specs.append(spec)

    # Pre-compute derived columns into working df
    work = _prepare_working_df(df, all_simple_specs)

    result = pd.DataFrame(index=df.index)

    for feat_str, spec in parsed:
        if isinstance(spec, InteractionSpec):
            left_series = _compute_single(spec.left, work)
            right_series = _compute_single(spec.right, work)
            if isinstance(left_series, pd.DataFrame) or isinstance(right_series, pd.DataFrame):
                raise ValueError(f"Cannot use Fourier features in interaction: {feat_str}")
            result[feat_str] = compute_interaction(left_series, right_series)
        else:
            computed = _compute_single(spec, work)
            if isinstance(computed, pd.DataFrame):
                for col in computed.columns:
                    result[col] = computed[col]
            else:
                result[feat_str] = computed

    logger.info(f"Computed {len(result.columns)} feature columns from {len(feature_list)} specs")
    return result


def extend_features(
    existing: pd.DataFrame,
    df: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """Extend an existing feature matrix with new rows.

    Computes features for dates beyond the existing dataset's last timestamp,
    then appends. Used for incremental updates — loads previous dataset from
    MLflow, computes only the new period.
    """
    last_ts = existing.index[-1]
    new_data = df.loc[df.index > last_ts]

    if new_data.empty:
        logger.info("No new data to extend")
        return existing

    lookback = pd.Timedelta(days=30)
    window_start = last_ts - lookback
    compute_data = df.loc[df.index >= window_start]

    new_features = engineer_features(compute_data, feature_list, validate=False)

    new_rows = new_features.loc[new_features.index > last_ts]

    combined = pd.concat([existing, new_rows])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    logger.info(f"Extended features: {len(existing)} + {len(new_rows)} = {len(combined)} rows")
    return combined
