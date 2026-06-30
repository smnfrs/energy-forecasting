"""Market feature functions (spreads, EWMA, rolling, lags, temporal).

Pure functions — each takes a DataFrame/Series + parameters, returns computed columns.
No sklearn transformers, no column mutation.

Ported from:
- EP: src/features/transforms.py (spreads, net exports, generation %, temporal, holidays)
- EP: src/features/ts_transforms.py (rolling stats, EWMA, lags, daily aggregates)
- EMA: data_modules/feature_eng.py (create_time_features)
- EMA: data_modules/data_classes.py (Fourier features)
"""

from datetime import date

import holidays
import numpy as np
import pandas as pd

from energy_forecasting.config.features import (
    CYCLICAL_PERIODS,
    DAY_INDEX_EPOCH,
    GENERATION_COLUMNS,
    GERMAN_STATE_POPULATIONS,
    NEIGHBOUR_PRICES,
    RENEWABLE_COLUMNS,
    YEAR_INDEX_BASE,
)

# ── Price spreads ─────────────────────────────────────────────────


def compute_price_spreads(
    df: pd.DataFrame,
    neighbours: list[str] | None = None,
) -> pd.DataFrame:
    """spread_{country} = target_price - neighbour_price.

    Positive spread = DE-LU more expensive than neighbour.
    """
    if neighbours is None:
        neighbours = NEIGHBOUR_PRICES
    target = df["target_price"]
    result = pd.DataFrame(index=df.index)
    for col in neighbours:
        # Extract country suffix from column name (e.g. marktpreis_frankreich -> frankreich)
        country = col.replace("marktpreis_", "")
        result[f"_derived_spread_{country}"] = target - df[col]
    return result


# ── Net exports ───────────────────────────────────────────────────


def compute_net_exports(
    df: pd.DataFrame,
    flow_pairs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """net_export_{country} = exports - imports.

    Also computes total_exports, total_imports.
    """
    if flow_pairs is None:
        from energy_forecasting.config.features import FLOW_PAIRS

        flow_pairs = FLOW_PAIRS

    result = pd.DataFrame(index=df.index)
    total_exp = pd.Series(0.0, index=df.index)
    total_imp = pd.Series(0.0, index=df.index)

    for export_col, import_col, country in flow_pairs:
        if export_col in df.columns and import_col in df.columns:
            exp = df[export_col].fillna(0)
            imp = df[import_col].fillna(0)
            result[f"_derived_net_export_{country}"] = exp - imp
            total_exp += exp
            total_imp += imp

    result["_derived_total_exports"] = total_exp
    result["_derived_total_imports"] = total_imp
    return result


# ── Generation percentages ───────────────────────────────────────


def compute_generation_pct(
    df: pd.DataFrame,
    sources: list[str] | None = None,
    add_renewable_pct: bool = False,
    add_supply_demand_gap: bool = False,
    add_prognosticated_pct: bool = False,
) -> pd.DataFrame:
    """pct_{source} = source / total_generation."""
    if sources is None:
        sources = GENERATION_COLUMNS

    total = df[sources].sum(axis=1)
    result = pd.DataFrame(index=df.index)
    result["_derived_total_generation"] = total

    for col in sources:
        pct_name = col.replace("stromerzeugung_", "_derived_gen_pct_")
        result[pct_name] = df[col] / total

    if add_renewable_pct:
        renewable = df[[c for c in RENEWABLE_COLUMNS if c in df.columns]].sum(axis=1)
        result["_derived_pct_renewable"] = renewable / total

    if add_supply_demand_gap:
        load_col = "stromverbrauch_gesamt_(netzlast)"
        if load_col in df.columns:
            result["_derived_supply_demand_gap"] = total - df[load_col]

    if add_prognosticated_pct:
        prog_total = "prognostizierte_erzeugung_gesamt"
        prog_other = "prognostizierte_erzeugung_sonstige"
        prog_wind_pv = "prognostizierte_erzeugung_wind_und_photovoltaik"
        prog_wind_on = "prognostizierte_erzeugung_onshore"
        prog_wind_off = "prognostizierte_erzeugung_offshore"
        prog_solar = "prognostizierte_erzeugung_photovoltaik"
        if prog_total in df.columns:
            pt = df[prog_total]
            if prog_other in df.columns:
                result["_derived_pct_prog_other"] = df[prog_other] / pt
            if prog_wind_pv in df.columns:
                result["_derived_pct_prog_wind_pv"] = df[prog_wind_pv] / pt
            if prog_wind_on in df.columns:
                result["_derived_pct_prog_wind_on"] = df[prog_wind_on] / pt
            if prog_wind_off in df.columns:
                result["_derived_pct_prog_wind_off"] = df[prog_wind_off] / pt
            if prog_solar in df.columns:
                result["_derived_pct_prog_solar"] = df[prog_solar] / pt

    return result


# ── EEG regime indicator ─────────────────────────────────────────


def compute_eeg_regime(index: pd.DatetimeIndex) -> pd.Series:
    """Categorical regime indicator for EEG §51 negative-price clawback rules.

    Returns a Series of integers per row:
        0 = pre-2023 (no clawback active)
        1 = 4h threshold (from EEG_4H_RULE_DATE)
        2 = 2h threshold (from EEG_2H_RULE_DATE)
        3 = Solarspitzengesetz, any negative 15-min block (from EEG_SOLARSPITZENGESETZ_DATE)
    """
    from energy_forecasting.config.modeling import EEG_REGIME_DATES

    tz = index.tz
    values = np.zeros(len(index), dtype=int)
    for date_str, regime in EEG_REGIME_DATES:
        threshold = pd.Timestamp(date_str, tz=tz) if tz is not None else pd.Timestamp(date_str)
        values = np.where(index >= threshold, regime, values)
    return pd.Series(values, index=index, dtype=int)


# ── Negative-price rolling statistics ────────────────────────────


def compute_neg_price_stats(price: pd.Series) -> pd.DataFrame:
    """Naive (un-lagged) rolling stats on the target price for negative-hour signal.

    Returns three columns:
        _derived_neg_price_frac_30d  — fraction of hours with price < 0 in last 30 days
        _derived_neg_price_frac_90d  — same, 90 days
        _derived_neg_price_depth_30d — average positive below-zero magnitude
                                       in last 30 days

    The engine applies the D-1 lag via the `_d1` suffix on the SHORT_NAME alias,
    so these series are stored un-lagged.
    """
    neg = (price < 0).astype(float)
    neg_depth = (-price).clip(lower=0)

    window_30d = 30 * 24
    window_90d = 90 * 24

    result = pd.DataFrame(index=price.index)
    result["_derived_neg_price_frac_30d"] = neg.rolling(window_30d, min_periods=1).mean()
    result["_derived_neg_price_frac_90d"] = neg.rolling(window_90d, min_periods=1).mean()
    result["_derived_neg_price_depth_30d"] = neg_depth.rolling(window_30d, min_periods=1).mean()
    return result


# ── Rolling statistics ────────────────────────────────────────────


def compute_rolling_stat(
    series: pd.Series,
    start_day: int,
    end_day: int,
    stat: str = "avg",
    end_hour: int | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
) -> pd.Series:
    """Rolling statistic over a day-relative historical window.

    For each row at hour H on day D:
    - Window = all rows from day D+start_day through D+end_day
      (start_day and end_day are negative, e.g. -7, -1)
    - end_hour: truncate the final day at this hour (hours 0 to end_hour-1)
    - hour_start/hour_end: filter to hours [hour_start, hour_end) on every day
    - stat: "avg", "std", "min", "max", "sum", "range"

    Special case: start_day=0, end_day=0 → current-day broadcast aggregate
    (groups by date, computes stat, broadcasts to all 24 hours).
    """
    if start_day == 0 and end_day == 0:
        return _compute_daily_broadcast(series, stat)

    # Fast path: simple multi-day window, no hour filter or end_hour
    if end_hour is None and hour_start is None:
        return _rolling_stat_fast(series, start_day, end_day, stat)

    # Slow path: complex windows with end_hour or hour filter
    return _rolling_stat_slow(series, start_day, end_day, stat, end_hour, hour_start, hour_end)


def _rolling_stat_fast(
    series: pd.Series,
    start_day: int,
    end_day: int,
    stat: str,
) -> pd.Series:
    """Fast rolling stat using daily pre-aggregation.

    Strategy:
    1. For each calendar day, collect all hourly values into a daily bucket
    2. Compute the stat across the appropriate range of daily buckets
    3. Broadcast back to hourly
    """
    # Step 1: compute per-day "buckets" of all values
    grouped = series.groupby(series.index.normalize())
    agg_func = _stat_to_pandas(stat)

    if start_day == end_day:
        # Single-day: compute stat within that day, shift by |offset| days
        if stat == "range":
            daily_max = grouped.max()
            daily_min = grouped.min()
            daily_stat = (daily_max - daily_min).shift(-start_day)
        else:
            daily_stat = grouped.agg(agg_func).shift(-start_day)
        vals = daily_stat.reindex(series.index.normalize())
        return pd.Series(vals.values, index=series.index, dtype=float)

    # Multi-day window: aggregate to daily, apply rolling, broadcast back.
    # avg/sum use daily pre-aggregation (exact for uniform 24h days).
    # std uses hourly rolling for exact results.
    # min/max/range use daily extremes (preserves semantics).
    n_days = abs(start_day - end_day) + 1
    day_shift = -end_day

    if stat == "avg":
        daily_mean = grouped.mean()
        rolled = daily_mean.rolling(window=n_days, min_periods=1).mean().shift(day_shift)
    elif stat == "sum":
        daily_sum = grouped.sum()
        rolled = daily_sum.rolling(window=n_days, min_periods=1).sum().shift(day_shift)
    elif stat in ("std", "min", "max", "range"):
        # For min/max/range, daily pre-aggregation preserves semantics.
        # For std, use hourly rolling (n_days*24 hours) to get exact result.
        if stat == "std":
            n_hours = n_days * 24
            # Compute hourly rolling std, pick the last hour of each day,
            # then shift by day_shift days
            hourly_std = series.rolling(window=n_hours, min_periods=2).std()
            # Take value at hour 23 of each day (complete window)
            daily_std = hourly_std[hourly_std.index.hour == 23]
            daily_std = daily_std.copy()
            daily_std.index = daily_std.index.normalize()
            rolled = daily_std.shift(day_shift)
        elif stat == "min":
            daily_min = grouped.min()
            rolled = daily_min.rolling(window=n_days, min_periods=1).min().shift(day_shift)
        elif stat == "max":
            daily_max = grouped.max()
            rolled = daily_max.rolling(window=n_days, min_periods=1).max().shift(day_shift)
        elif stat == "range":
            daily_max = grouped.max()
            daily_min = grouped.min()
            r_max = daily_max.rolling(window=n_days, min_periods=1).max().shift(day_shift)
            r_min = daily_min.rolling(window=n_days, min_periods=1).min().shift(day_shift)
            rolled = r_max - r_min
    else:
        raise ValueError(f"Unknown stat {stat!r}")

    vals = rolled.reindex(series.index.normalize())
    return pd.Series(vals.values, index=series.index, dtype=float)


def _rolling_stat_slow(
    series: pd.Series,
    start_day: int,
    end_day: int,
    stat: str,
    end_hour: int | None,
    hour_start: int | None,
    hour_end: int | None,
) -> pd.Series:
    """Slow per-day loop for complex windows (end_hour or hour filter)."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    dates = series.index.normalize().unique()

    for current_date in dates:
        window_start = current_date + pd.Timedelta(days=start_day)
        window_end_date = current_date + pd.Timedelta(days=end_day)

        if end_hour is not None:
            full_days_end = window_end_date - pd.Timedelta(days=1)
            mask_full = (series.index >= window_start) & (
                series.index < full_days_end + pd.Timedelta(days=1)
            )
            mask_last = (series.index >= window_end_date) & (
                series.index < window_end_date + pd.Timedelta(hours=end_hour)
            )
            mask = mask_full | mask_last
        else:
            mask = (series.index >= window_start) & (
                series.index < window_end_date + pd.Timedelta(days=1)
            )

        if hour_start is not None and hour_end is not None:
            hour_mask = (series.index.hour >= hour_start) & (series.index.hour < hour_end)
            mask = mask & hour_mask

        window_data = series.loc[mask].dropna()

        if len(window_data) == 0:
            continue

        val = _apply_stat(window_data, stat)

        day_mask = series.index.normalize() == current_date
        result.loc[day_mask] = val

    return result


def _compute_daily_broadcast(series: pd.Series, stat: str) -> pd.Series:
    """Group by date, compute stat, broadcast to all hours."""
    grouped = series.groupby(series.index.normalize())

    if stat == "share":
        daily_sum = grouped.transform("sum")
        return series / daily_sum

    agg_func = _stat_to_pandas(stat)
    return grouped.transform(agg_func)


def _apply_stat(data: pd.Series, stat: str) -> float:
    """Apply a statistic to a series."""
    if stat == "avg":
        return data.mean()
    elif stat == "std":
        return data.std()
    elif stat == "min":
        return data.min()
    elif stat == "max":
        return data.max()
    elif stat == "sum":
        return data.sum()
    elif stat == "range":
        return data.max() - data.min()
    else:
        raise ValueError(f"Unknown stat {stat!r}")


def _stat_to_pandas(stat: str) -> str:
    """Map our stat names to pandas aggregation function names."""
    return {"avg": "mean", "std": "std", "min": "min", "max": "max", "sum": "sum"}.get(stat, stat)


# ── EWMA with information cutoff ─────────────────────────────────


def compute_ewma(
    series: pd.Series,
    span: int,
    cutoff_day: int | None = None,
    cutoff_hour: int | None = None,
) -> pd.Series:
    """EWMA with information cutoff boundary.

    - cutoff_day=-1 → use data up to end of yesterday
    - cutoff_day=-1, cutoff_hour=10 → use data up to yesterday 10:00
    - cutoff_day=-2 → use data up to end of two days ago

    EWMA value at the cutoff is broadcast to all hours of the prediction day.
    """
    if cutoff_day is None:
        return series.ewm(span=span, min_periods=1).mean()

    # Pre-compute full EWMA
    full_ewma = series.ewm(span=span, min_periods=1).mean()

    if cutoff_hour is None:
        cutoff_hour = 23

    # The cutoff timestamp for day D is: (D + cutoff_day) at cutoff_hour.
    # We need the EWMA value at that timestamp, broadcast to all hours of D.
    # Strategy: for each day, look up the EWMA at the cutoff position.

    # Get the EWMA at the end of each day (hour 23)
    if cutoff_hour == 23:
        # Simple case: use last value of each day, then shift by |cutoff_day| days
        daily_last = full_ewma.resample("D").last()
        shifted = daily_last.shift(-cutoff_day)
    else:
        # Need the EWMA value at a specific hour of day
        # Filter to just the cutoff hour, then shift
        hour_mask = full_ewma.index.hour == cutoff_hour
        at_cutoff_hour = full_ewma[hour_mask]
        # This is one value per day — shift by |cutoff_day| days
        shifted = at_cutoff_hour.shift(-cutoff_day)
        # Align back to daily index
        shifted.index = shifted.index.normalize()

    # Broadcast daily values to hourly
    result = shifted.reindex(series.index.normalize())
    return pd.Series(result.values, index=series.index, dtype=float)


# ── Hourly lag ────────────────────────────────────────────────────


def compute_hourly_lag(series: pd.Series, hours: int) -> pd.Series:
    """Value N hours back: series.shift(hours)."""
    return series.shift(hours)


# ── Temporal features ─────────────────────────────────────────────


def compute_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar and cyclical time features.

    Returns columns: hour_sin, hour_cos, dow_sin, dow_cos,
    month_sin, month_cos, is_weekend, is_holiday, day_index, year_index.
    """
    result = pd.DataFrame(index=index)

    hour = index.hour
    dow = index.dayofweek
    month = index.month

    # Cyclical encoding
    for name, period in CYCLICAL_PERIODS.items():
        if name == "hour":
            values = hour
            prefix = "hour"
        elif name == "day_of_week":
            values = dow
            prefix = "dow"
        elif name == "month":
            values = month
            prefix = "month"
        else:
            continue

        result[f"_derived_{prefix}_sin"] = np.sin(2 * np.pi * values / period)
        result[f"_derived_{prefix}_cos"] = np.cos(2 * np.pi * values / period)

    result["_derived_is_weekend"] = (dow >= 5).astype(float)
    result["_derived_is_holiday"] = compute_german_holidays(index)
    epoch = pd.Timestamp(DAY_INDEX_EPOCH, tz=index.tz)
    result["_derived_day_index"] = (index.normalize() - epoch).days
    result["_derived_year_index"] = index.year - YEAR_INDEX_BASE

    return result


def compute_german_holidays(index: pd.DatetimeIndex) -> pd.Series:
    """Population-weighted German holiday indicator (0.0 to 1.0).

    National holidays → 1.0, state-specific → fraction of population observing.
    """
    total_pop = sum(GERMAN_STATE_POPULATIONS.values())
    years = sorted(set(index.year))

    # Build state holiday sets
    state_holidays: dict[str, set[date]] = {}
    for state_code in GERMAN_STATE_POPULATIONS:
        state_set: set[date] = set()
        for yr in years:
            state_set.update(holidays.Germany(subdiv=state_code, years=yr).keys())
        state_holidays[state_code] = state_set

    # For each date, compute weighted fraction
    unique_dates = index.normalize().unique()
    date_weights: dict[pd.Timestamp, float] = {}
    for ts in unique_dates:
        d = ts.date()
        weight = 0.0
        for state_code, pop in GERMAN_STATE_POPULATIONS.items():
            if d in state_holidays[state_code]:
                weight += pop
        date_weights[ts] = weight / total_pop

    return pd.Series(
        [date_weights.get(ts, 0.0) for ts in index.normalize()],
        index=index,
        dtype=float,
    )


# ── Fourier features ─────────────────────────────────────────────


def compute_fourier_features(
    index: pd.DatetimeIndex,
    period: int,
    order: int,
) -> pd.DataFrame:
    """Deterministic Fourier terms for arbitrary-period seasonality.

    For order=O, produces 2*O columns:
      sin_1, cos_1, sin_2, cos_2, ..., sin_O, cos_O
    where sin_k = sin(2pi * k * t / period), t = integer position in index.
    """
    t = np.arange(len(index), dtype=float)
    result = pd.DataFrame(index=index)
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        result[f"fourier_{period}_{order}_sin_{k}"] = np.sin(angle)
        result[f"fourier_{period}_{order}_cos_{k}"] = np.cos(angle)
    return result


# ── Trend and interaction features ────────────────────────────────


def compute_day_index(index: pd.DatetimeIndex) -> pd.Series:
    """Days since DAY_INDEX_EPOCH (2015-01-05). Integer."""
    epoch = pd.Timestamp(DAY_INDEX_EPOCH, tz=index.tz)
    return pd.Series(
        (index.normalize() - epoch).days,
        index=index,
        dtype=int,
    )


def compute_interaction(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise product: left * right."""
    return left * right
