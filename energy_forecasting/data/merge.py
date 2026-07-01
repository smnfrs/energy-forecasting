"""Data cleaning and merge pipeline.

Orchestrates loading raw Parquet files, combining national SMARD eras,
creating the unified target price, merging commodities, enforcing
periodicity, imputing gaps, running cleaning rules, normalizing DST,
and validating the output.

Produces:
    data/processed/merged.parquet   -- national dataset (local time)
    data/processed/tso/*.parquet    -- per-TSO cleaned data (UTC)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from energy_forecasting.config.merge import (
    BIDDING_AREA_SPLIT,
    IMPUTE_WINDOW_DAYS,
    MAX_MISSING_CONSECUTIVE,
    MEDIUM_GAP_MAX,
    PRICE_POST_SPLIT,
    PRICE_PRE_SPLIT,
    QUARTER_HOURLY_START,
    SMALL_GAP_MAX,
    SMARD_WARN_BOUNDS,
    TSO_TO_NATIONAL,
)
from energy_forecasting.data.io import load_parquet, save_parquet
from energy_forecasting.data.processing import _match_columns

# -- Physical bounds warnings -----------------------------------------------


def warn_physical_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> None:
    """Log warnings for SMARD values outside expected physical ranges.

    Pre-cleaning diagnostic -- does NOT modify the data.
    """
    if bounds is None:
        bounds = SMARD_WARN_BOUNDS

    for pattern, (lo, hi) in bounds.items():
        cols = _match_columns(df, pattern)
        for col in cols:
            series = df[col].dropna()
            if series.empty:
                continue
            violations = 0
            if lo is not None:
                violations += (series < lo).sum()
            if hi is not None:
                violations += (series > hi).sum()
            if violations > 0:
                logger.warning(
                    f"Physical bounds: {col} has {violations} values "
                    f"outside [{lo}, {hi}] "
                    f"(range: {series.min():.1f} to {series.max():.1f})"
                )


# -- Periodicity enforcement ------------------------------------------------


def enforce_periodicity(
    df: pd.DataFrame,
    max_gap: int = MAX_MISSING_CONSECUTIVE,
) -> pd.DataFrame:
    """Detect and fill small gaps in the hourly timestamp index.

    Ported from EMA's fix_broken_periodicity_with_interpolation().
    """
    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    missing = expected.difference(df.index)

    if missing.empty:
        return df

    # Group consecutive missing timestamps into runs
    missing_series = pd.Series(missing, dtype="datetime64[ns, UTC]")
    breaks = missing_series.diff() != pd.Timedelta(hours=1)
    groups = breaks.cumsum()
    run_sizes = groups.value_counts()

    too_large = run_sizes[run_sizes > max_gap]
    if not too_large.empty:
        # Find the first offending run for a useful error message
        first_group = too_large.index[0]
        first_ts = missing_series[groups == first_group].iloc[0]
        raise ValueError(
            f"Gap of {too_large.iloc[0]} consecutive missing hours "
            f"starting at {first_ts} exceeds max_gap={max_gap}"
        )

    logger.info(f"Periodicity: filling {len(missing)} missing timestamps")
    df = df.reindex(expected)
    df = df.interpolate(method="time")
    return df


# -- Medium-gap imputation --------------------------------------------------


def impute_medium_gaps(
    df: pd.DataFrame,
    small_gap_max: int = SMALL_GAP_MAX,
    medium_gap_max: int = MEDIUM_GAP_MAX,
    window_days: int = IMPUTE_WINDOW_DAYS,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Fill medium NaN gaps (6-48h) using same-hour-of-day averaging."""
    exclude_set = set(exclude or [])
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_set]

    for col in numeric_cols:
        series = df[col]
        if not series.isna().any():
            continue

        is_na = series.isna()
        gap_groups = is_na.ne(is_na.shift()).cumsum()
        gap_sizes = is_na.groupby(gap_groups).transform("sum")

        medium_mask = is_na & (gap_sizes > small_gap_max) & (gap_sizes <= medium_gap_max)

        # Log large gaps (always, regardless of medium gaps)
        large_mask = is_na & (gap_sizes > medium_gap_max)
        if large_mask.any():
            for g in gap_groups[large_mask].unique():
                g_idx = df.index[gap_groups == g]
                logger.warning(
                    f"Large gap ({len(g_idx)}h) in {col} starting {g_idx[0]} -- not imputed"
                )

        if not medium_mask.any():
            continue

        # Fill medium gaps with same-hour-of-day average
        window = pd.Timedelta(days=window_days)
        positions = df.index[medium_mask]
        for ts in positions:
            hour = ts.hour
            window_start = ts - window
            window_end = ts + window
            window_data = series.loc[window_start:window_end]
            same_hour = window_data[window_data.index.hour == hour].dropna()
            if not same_hour.empty:
                df.loc[ts, col] = same_hour.mean()

    return df


# -- National SMARD merge ---------------------------------------------------


def merge_national_smard(
    df_de_lu: pd.DataFrame,
    df_de_at_lu: pd.DataFrame,
    cutoff: pd.Timestamp = BIDDING_AREA_SPLIT,
) -> pd.DataFrame:
    """Split and concatenate the two national SMARD datasets at the cutoff.

    Ported from EP's merge_datasets().
    """
    pre = df_de_at_lu[df_de_at_lu.index < cutoff]
    post = df_de_lu[df_de_lu.index >= cutoff]
    merged = pd.concat([pre, post], axis=0)
    merged = merged.sort_index()
    return merged


# -- Unified target price ---------------------------------------------------


def create_unified_target(
    df: pd.DataFrame,
    ec_fallback: pd.Series | None = None,
) -> pd.DataFrame:
    """Create target_price from post-split and pre-split price columns.

    Priority: post-split > pre-split > Energy Charts fallback.
    """
    post = df.get(PRICE_POST_SPLIT)
    pre = df.get(PRICE_PRE_SPLIT)

    if post is not None and pre is not None:
        target = post.combine_first(pre)
    elif post is not None:
        target = post
    elif pre is not None:
        target = pre
    else:
        target = pd.Series(float("nan"), index=df.index, name="target_price")

    if ec_fallback is not None:
        ec_aligned = ec_fallback.reindex(target.index)
        gaps_before = target.isna().sum()
        target = target.combine_first(ec_aligned)
        gaps_filled = gaps_before - target.isna().sum()
        if gaps_filled > 0:
            logger.info(f"Energy Charts filled {gaps_filled} target_price gaps")

    df["target_price"] = target
    return df


# -- Energy Charts extension ------------------------------------------------


def extend_with_energy_charts(
    df: pd.DataFrame,
    ec_fallback: pd.Series | None,
) -> pd.DataFrame:
    """Append Energy Charts rows beyond the last SMARD timestamp."""
    if ec_fallback is None or ec_fallback.empty:
        return df

    last_smard = df.index.max()
    ec_beyond = ec_fallback[ec_fallback.index > last_smard]

    if ec_beyond.empty:
        return df

    # Resample quarter-hourly to hourly if needed
    freq = pd.infer_freq(ec_beyond.index)
    if freq is not None and "15" in str(freq):
        ec_beyond = ec_beyond.resample("h").mean()

    # Create new rows with only target_price filled
    new_rows = pd.DataFrame(index=ec_beyond.index, columns=df.columns)
    new_rows["target_price"] = ec_beyond.values

    df = pd.concat([df, new_rows])
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    logger.info(f"Extended {len(ec_beyond)} rows with Energy Charts data")
    return df


# -- Regime indicators ------------------------------------------------------


def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary regime indicator columns."""
    df["regime_de_at_lu"] = (df.index < BIDDING_AREA_SPLIT).astype(int)
    df["regime_quarter_hourly"] = (df.index >= QUARTER_HOURLY_START).astype(int)
    return df


# -- Commodity construction and merge ---------------------------------------


def build_commodity_daily(raw_dir: Path) -> pd.DataFrame:
    """Build a daily commodity price DataFrame from raw Parquet files."""
    from energy_forecasting.data.commodities import merge_carbon, reconstruct_ttf

    frames = []

    ttf = reconstruct_ttf(raw_dir)
    frames.append(ttf)

    carbon = merge_carbon(raw_dir)
    frames.append(carbon)

    brent_path = raw_dir / "brent.parquet"
    if brent_path.exists():
        brent = load_parquet(brent_path)
        from energy_forecasting.config.commodities import COLUMN_NAMES

        brent_series = brent["price"].rename(COLUMN_NAMES["brent"])
        frames.append(brent_series)

    daily = pd.concat(frames, axis=1)
    daily = daily.sort_index()
    return daily


# Max calendar days to forward-fill commodity gaps. Commodity markets close for
# extended holidays (Christmas/New Year ~3 weeks for ICAP carbon). Must be large
# enough to cover these, but small enough to catch genuine pipeline failures
# (which would show gaps of months, not weeks).
MAX_COMMODITY_FFILL_DAYS = 30


def merge_commodities(
    df: pd.DataFrame,
    commodity_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Join daily commodity prices onto the hourly dataset via forward-fill.

    Daily commodity data only has rows for business days (no weekends/holidays).
    We reindex to a complete calendar-day range and ffill within a bounded
    window (MAX_COMMODITY_FFILL_DAYS) to cover weekends and holiday closures
    without masking genuine data pipeline failures.
    """
    # Reindex daily data to complete calendar range, then bounded ffill.
    # This fills weekends (2 days) and holiday closures (up to ~5 biz days)
    # but won't silently cover a broken download (which would show gaps >> 7 days).
    full_daily_idx = pd.date_range(
        commodity_daily.index.min(), commodity_daily.index.max(), freq="D"
    )
    commodity_filled = commodity_daily.reindex(full_daily_idx).ffill(limit=MAX_COMMODITY_FFILL_DAYS)

    # Map each hourly timestamp to its calendar date, then look up
    dates = df.index.normalize()
    commodity_hourly = commodity_filled.reindex(dates)
    commodity_hourly.index = df.index

    for col in commodity_hourly.columns:
        df[col] = commodity_hourly[col]

    return df


# -- DST normalization ------------------------------------------------------


def normalize_dst(
    df: pd.DataFrame,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Normalize DST transitions to ensure exactly 24 hours per local day.

    Converts UTC index to naive local time (wall clock), then:
    - Spring-forward (23h days): interpolates the missing hour
    - Fall-back (25h days): averages the duplicate hour

    Output index is tz-naive representing local delivery hours (0-23).
    The nonexistent spring-forward hour (e.g., 02:00 CET on March 29)
    cannot be represented as a tz-aware Europe/Berlin timestamp, so
    tz-naive is the correct representation for normalized delivery hours.

    Ported from EP's normalize_dst(), with the spring-forward timestamp
    bug fixed (EP used timedelta arithmetic which produces hour 3 instead
    of hour 2 due to UTC-based arithmetic on tz-aware timestamps).
    """
    df = df.copy()

    # Convert to local time, then strip tz for manipulation.
    # After tz_localize(None), spring-forward days have a gap (no 02:00)
    # and fall-back days have duplicates (two 02:00 rows).
    df.index = df.index.tz_convert(timezone).tz_localize(None)

    numeric_cols = df.select_dtypes(include="number").columns

    # Collect modifications (don't mutate df during iteration)
    to_insert: list[pd.DataFrame] = []
    to_drop: list[pd.Timestamp] = []
    to_add: list[pd.DataFrame] = []

    for day, day_df in df.groupby(df.index.date):
        n = len(day_df)

        if n == 23:
            # Spring-forward: find and interpolate missing hour
            present = set(day_df.index.hour)
            missing = set(range(24)) - present
            if not missing:
                continue

            missing_hour = min(missing)
            prev_rows = day_df[day_df.index.hour == missing_hour - 1]
            next_rows = day_df[day_df.index.hour == missing_hour + 1]

            if prev_rows.empty or next_rows.empty:
                continue

            prev_row = prev_rows.iloc[0]
            next_row = next_rows.iloc[0]
            new_row = prev_row.copy()
            for col in numeric_cols:
                if pd.notna(prev_row[col]) and pd.notna(next_row[col]):
                    new_row[col] = (prev_row[col] + next_row[col]) / 2

            # Create naive local timestamp for the missing hour
            new_ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=missing_hour)
            to_insert.append(pd.DataFrame([new_row], index=[new_ts]))

        elif n == 25:
            # Fall-back: average duplicate naive timestamps
            dup_mask = day_df.index.duplicated(keep=False)
            if not dup_mask.any():
                continue

            dup_ts = day_df.index[dup_mask][0]  # the duplicated timestamp
            dup_rows = day_df.loc[[dup_ts]]

            averaged = dup_rows.iloc[0].copy()
            for col in numeric_cols:
                vals = dup_rows[col].dropna()
                if not vals.empty:
                    averaged[col] = vals.mean()

            # Drop all instances, add back one averaged row
            to_drop.append(dup_ts)
            to_add.append(pd.DataFrame([averaged], index=[dup_ts]))

    # Apply all changes at once
    if to_drop:
        # drop by label removes all rows with that label (both duplicates)
        df = df.drop(to_drop)

    if to_add or to_insert:
        df = pd.concat([df] + to_add + to_insert)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# -- NaN validation gate ----------------------------------------------------


DEFAULT_CRITICAL_COLS = [
    "target_price",
    "stromverbrauch_gesamt_(netzlast)",
    "stromerzeugung_wind_onshore",
    "stromerzeugung_photovoltaik",
]


def validate_no_nans(
    df: pd.DataFrame,
    critical_cols: list[str] | None = None,
    *,
    allow_initial_target_gap: bool = False,
) -> None:
    """Post-cleaning validation for NaN gaps.

    Critical columns fail fast. ``target_price`` can optionally tolerate a
    contiguous start-of-series gap, but internal and trailing gaps always fail.
    Noncritical NaNs remain warnings.
    """
    if critical_cols is None:
        critical_cols = DEFAULT_CRITICAL_COLS

    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if nan_cols.empty:
        logger.info("NaN gate: all columns clean")
        return

    for col, count in nan_cols.items():
        pct = count / len(df) * 100
        if col in critical_cols:
            if col == "target_price" and _target_price_gap_allowed(
                df[col],
                allow_initial_gap=allow_initial_target_gap,
            ):
                continue
            raise ValueError(f"NaN gate: CRITICAL {col} has {count} NaN ({pct:.1f}%)")
        else:
            logger.warning(f"NaN gate: {col} has {count} NaN ({pct:.1f}%)")


def _target_price_gap_allowed(
    target: pd.Series,
    *,
    allow_initial_gap: bool,
) -> bool:
    """Return True only for an explicitly allowed initial target gap."""
    missing = target.isna()
    if not missing.any():
        return True

    first_valid_idx = target.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("NaN gate: target_price is entirely NaN")

    first_valid_loc = target.index.get_loc(first_valid_idx)
    if isinstance(first_valid_loc, slice):
        first_valid_loc = first_valid_loc.start
    else:
        first_valid_loc = int(first_valid_loc)

    prefix_only = bool(
        missing.iloc[:first_valid_loc].all()
        and not missing.iloc[first_valid_loc:].any()
    )
    if prefix_only and allow_initial_gap:
        logger.warning(
            f"NaN gate: target_price has allowed initial prefix gap "
            f"({int(missing.sum())} rows)"
        )
        return True

    if prefix_only:
        raise ValueError(
            "NaN gate: target_price has an initial gap; pass "
            "allow_initial_target_gap=True only when this is expected"
        )
    raise ValueError("NaN gate: target_price has internal or trailing NaN gaps")


# -- Cross-validation: national vs per-TSO ----------------------------------


def cross_validate_national_vs_tso(
    national_path: Path,
    tso_dir: Path,
    mapping: dict[str, str] | None = None,
) -> None:
    """Compare national generation/load totals against sum-of-TSO."""
    if mapping is None:
        mapping = TSO_TO_NATIONAL

    from energy_forecasting.config.smard import TSO_REGIONS, TSO_SUFFIXES

    if not national_path.exists():
        logger.warning("Cross-validation: national file not found")
        return

    national = load_parquet(national_path)

    # Load all TSO files
    tso_data: dict[str, pd.DataFrame] = {}
    for tso_name in TSO_REGIONS:
        tso_path = tso_dir / f"{tso_name}.parquet"
        if tso_path.exists():
            tso_data[tso_name] = load_parquet(tso_path)

    if not tso_data:
        logger.warning("Cross-validation: no TSO files found")
        return

    for tso_base_name, national_col in mapping.items():
        if national_col not in national.columns:
            continue

        # Sum across all TSOs for this metric
        tso_series_list = []
        for tso_name, tso_df in tso_data.items():
            suffix = TSO_SUFFIXES[tso_name]
            tso_col = f"{tso_base_name}{suffix}"
            if tso_col in tso_df.columns:
                tso_series_list.append(tso_df[tso_col])

        if not tso_series_list:
            continue

        # Align to common index and sum
        tso_combined = pd.concat(tso_series_list, axis=1)
        tso_sum = tso_combined.sum(axis=1)

        # Align national to TSO index (TSO is UTC, national may be local)
        national_series = national[national_col]
        common = national_series.reindex(tso_sum.index).dropna()
        tso_common = tso_sum.reindex(common.index).dropna()

        if common.empty or tso_common.empty:
            continue

        # Compute on overlapping non-NaN
        overlap = common.index.intersection(tso_common.index)
        if overlap.empty:
            continue

        nat_vals = common.loc[overlap]
        tso_vals = tso_common.loc[overlap]

        # MAPE (skip zeros to avoid division)
        nonzero = nat_vals != 0
        if nonzero.any():
            mape = (
                (nat_vals[nonzero] - tso_vals[nonzero]).abs() / nat_vals[nonzero].abs()
            ).mean() * 100
            max_diff = (nat_vals - tso_vals).abs().max()
            logger.info(
                f"Cross-validate {tso_base_name}: "
                f"MAPE={mape:.1f}%, max_diff={max_diff:.0f} MW "
                f"({len(overlap)} common hours)"
            )


# -- Per-TSO cleaning -------------------------------------------------------


def clean_tso_data(
    tso_input_dir: Path,
    tso_output_dir: Path,
) -> None:
    """Clean all per-TSO SMARD Parquet files."""
    from energy_forecasting.config.smard import TSO_REGIONS
    from energy_forecasting.data.processing import interpolate_gaps

    tso_output_dir.mkdir(parents=True, exist_ok=True)

    for tso_name in TSO_REGIONS:
        input_path = tso_input_dir / f"{tso_name}.parquet"
        if not input_path.exists():
            logger.warning(f"TSO file not found: {input_path}")
            continue

        logger.info(f"Cleaning TSO: {tso_name}")
        df = load_parquet(input_path)

        # 1. Enforce periodicity
        try:
            df = enforce_periodicity(df)
        except ValueError as e:
            logger.error(f"TSO {tso_name} periodicity failed: {e}")
            continue

        # 2. Medium gap imputation
        df = impute_medium_gaps(df)

        # 3. Small gap interpolation
        df = interpolate_gaps(df, method="cubicspline", max_gap=SMALL_GAP_MAX)

        save_parquet(df.sort_index(), tso_output_dir / f"{tso_name}.parquet")
        logger.info(
            f"Cleaned {tso_name}: {len(df)} rows, "
            f"{len(df.columns)} columns, "
            f"NaN remaining: {df.isna().sum().sum()}"
        )


# -- Helpers -----------------------------------------------------------------


def _extract_ec_price(ec_df: pd.DataFrame) -> pd.Series | None:
    """Extract the price column from an Energy Charts DataFrame."""
    col = next(
        (c for c in ec_df.columns if "price" in c.lower()),
        ec_df.columns[0] if len(ec_df.columns) > 0 else None,
    )
    if col is None:
        return None
    return ec_df[col]


# -- Pipeline orchestrator ---------------------------------------------------


def run_merge_pipeline(
    smard_dir: Path | None = None,
    commodities_dir: Path | None = None,
    ec_dir: Path | None = None,
    output_path: Path | None = None,
    tso_output_dir: Path | None = None,
) -> pd.DataFrame:
    """Execute the full merge pipeline."""
    from energy_forecasting.config import (
        COMMODITIES_DIR,
        ENERGY_CHARTS_DIR,
        PROCESSED_DATA_DIR,
        SMARD_DIR,
    )
    from energy_forecasting.config.cleaning import clean

    smard_dir = smard_dir or SMARD_DIR
    commodities_dir = commodities_dir or COMMODITIES_DIR
    ec_dir = ec_dir or ENERGY_CHARTS_DIR
    output_path = output_path or PROCESSED_DATA_DIR / "merged.parquet"
    tso_output_dir = tso_output_dir or PROCESSED_DATA_DIR / "tso"

    # 1. Load national SMARD data
    de_lu_path = smard_dir / "DE-LU.parquet"
    de_at_lu_path = smard_dir / "DE-AT-LU.parquet"

    if not de_lu_path.exists() or not de_at_lu_path.exists():
        raise FileNotFoundError(
            f"National SMARD files not found. Run 'make data' first. "
            f"Expected: {de_lu_path}, {de_at_lu_path}"
        )

    logger.info("Loading national SMARD data")
    df_de_lu = load_parquet(de_lu_path)
    df_de_at_lu = load_parquet(de_at_lu_path)

    # 2. Physical bounds warnings (pre-cleaning diagnostic)
    warn_physical_bounds(df_de_lu)
    warn_physical_bounds(df_de_at_lu)

    # 3. Enforce periodicity on each dataset
    logger.info("Enforcing periodicity")
    df_de_lu = enforce_periodicity(df_de_lu)
    df_de_at_lu = enforce_periodicity(df_de_at_lu)

    # 4. Merge at bidding area split
    logger.info("Merging national SMARD at bidding area split")
    df = merge_national_smard(df_de_lu, df_de_at_lu)

    # 5. Load Energy Charts fallback and create unified target
    ec_fallback = None
    ec_path = ec_dir / "da_price_de_lu.parquet"
    if ec_path.exists():
        ec_fallback = _extract_ec_price(load_parquet(ec_path))

    logger.info("Creating unified target price")
    df = create_unified_target(df, ec_fallback)

    # 6. Extend with Energy Charts
    df = extend_with_energy_charts(df, ec_fallback)

    # 7. Regime indicators
    logger.info("Adding regime indicators")
    df = add_regime_indicators(df)

    # 8. Build and merge commodities
    logger.info("Merging commodity prices")
    try:
        commodity_daily = build_commodity_daily(commodities_dir)
        df = merge_commodities(df, commodity_daily)
    except FileNotFoundError as e:
        logger.warning(f"Commodity merge skipped: {e}")

    # 9. Medium gap imputation
    logger.info("Imputing medium gaps (same-hour-of-day)")
    df = impute_medium_gaps(
        df,
        exclude=["regime_de_at_lu", "regime_quarter_hourly", "target_price"],
    )

    # 10. Cleaning rules (structural fills + small gap interpolation)
    logger.info("Applying cleaning rules")
    df = clean(df)

    # 11. DST normalization
    logger.info("Normalizing DST -> Europe/Berlin")
    df = normalize_dst(df)

    # 12. NaN validation gate
    validate_no_nans(df, allow_initial_target_gap=True)

    # 13. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, output_path)
    logger.info(
        f"Saved merged dataset: {output_path.name}, "
        f"{len(df)} rows, {len(df.columns)} columns, "
        f"{df.index.min()} to {df.index.max()}"
    )

    # 14. Clean per-TSO data
    tso_input_dir = smard_dir / "tso"
    if tso_input_dir.exists():
        logger.info("Cleaning per-TSO data")
        clean_tso_data(tso_input_dir, tso_output_dir)

        # 15. Cross-validation
        cross_validate_national_vs_tso(output_path, tso_output_dir)
    else:
        logger.info("No per-TSO data found, skipping TSO cleaning")

    return df
