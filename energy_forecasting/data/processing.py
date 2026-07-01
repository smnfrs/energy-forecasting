"""Cleaning helper functions used by config/cleaning.py.

Each function implements one type of cleaning operation. The config module
calls these with domain-specific arguments. Full implementations in stage 3.
"""

from __future__ import annotations

from fnmatch import fnmatch

import numpy as np
import pandas as pd


def _match_columns(df: pd.DataFrame, pattern: str) -> list[str]:
    """Return column names matching an fnmatch pattern."""
    return [c for c in df.columns if fnmatch(c, pattern)]


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns if they exist, silently skip missing ones."""
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing)


def clip_bounds(
    df: pd.DataFrame,
    pattern: str,
    min_val: float | None = None,
    max_val: float | None = None,
    action: str = "clip",
) -> pd.DataFrame:
    """Clip or NaN values outside physical bounds for columns matching pattern.

    action='clip' clips to bounds; action='nan' sets out-of-bounds to NaN.
    """
    cols = _match_columns(df, pattern)
    for col in cols:
        if action == "nan":
            mask = pd.Series(False, index=df.index)
            if min_val is not None:
                mask |= df[col] < min_val
            if max_val is not None:
                mask |= df[col] > max_val
            df.loc[mask, col] = np.nan
        else:
            df[col] = df[col].clip(lower=min_val, upper=max_val)
    return df


def fill_zero_after(
    df: pd.DataFrame,
    columns: str | list[str],
    after: str,
) -> pd.DataFrame:
    """Fill NaN with 0 after a cutoff date (or after last valid observation)."""
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col not in df.columns:
            continue
        if after == "last_valid":
            last_idx = df[col].last_valid_index()
            if last_idx is not None:
                df.loc[df.index > last_idx, col] = df.loc[df.index > last_idx, col].fillna(0)
        else:
            cutoff = pd.Timestamp(after)
            df.loc[df.index > cutoff, col] = df.loc[df.index > cutoff, col].fillna(0)
    return df


def fill_zero_before(
    df: pd.DataFrame,
    columns: str | list[str],
    before: str,
) -> pd.DataFrame:
    """Fill NaN with 0 before a cutoff date."""
    if isinstance(columns, str):
        columns = [columns]
    cutoff = pd.Timestamp(before)
    for col in columns:
        if col not in df.columns:
            continue
        df.loc[df.index < cutoff, col] = df.loc[df.index < cutoff, col].fillna(0)
    return df


def fill_zero_before_first_valid(
    df: pd.DataFrame,
    columns: str | list[str],
) -> pd.DataFrame:
    """Fill NaN with 0 before the first valid observation."""
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col not in df.columns:
            continue
        first_idx = df[col].first_valid_index()
        if first_idx is not None:
            df.loc[df.index < first_idx, col] = df.loc[df.index < first_idx, col].fillna(0)
    return df


def fill_from_difference(
    df: pd.DataFrame,
    target: str,
    total: str,
    subtract: str,
) -> pd.DataFrame:
    """Fill NaN in target with total - subtract."""
    if target not in df.columns:
        return df
    mask = df[target].isna()
    if total in df.columns and subtract in df.columns:
        df.loc[mask, target] = df.loc[mask, total] - df.loc[mask, subtract]
    return df


def fill_from_column(
    df: pd.DataFrame,
    columns: str | list[str],
    source: str,
) -> pd.DataFrame:
    """Fill NaN in columns using values from source column."""
    if isinstance(columns, str):
        columns = [columns]
    if source not in df.columns:
        return df
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(df[source])
    return df


def fill_gen_total(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing generation total using component sum fallback.

    Uses sum of individual generation columns when the total is missing,
    but only if recent data (within 30 days) is available.
    """
    total_col = "stromerzeugung_gesamt"
    if total_col not in df.columns:
        return df

    gen_cols = [c for c in df.columns if c.startswith("stromerzeugung_") and c != total_col]
    mask = df[total_col].isna()
    if mask.any() and gen_cols:
        component_sum = df.loc[mask, gen_cols].sum(axis=1)
        # Only fill where we have at least some component data
        has_data = df.loc[mask, gen_cols].notna().any(axis=1)
        df.loc[mask & has_data.reindex(df.index, fill_value=False), total_col] = component_sum[
            has_data
        ]
    return df


def interpolate_gaps(
    df: pd.DataFrame,
    method: str = "cubicspline",
    max_gap: int = 5,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate small gaps in numeric columns.

    Only fills gaps of max_gap or fewer consecutive NaNs.
    """
    exclude = set(exclude or [])
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]

    for col in numeric_cols:
        series = df[col]
        if not series.isna().any():
            continue
        # Identify gap sizes: consecutive NaN groups
        is_na = series.isna()
        gap_groups = is_na.ne(is_na.shift()).cumsum()
        gap_sizes = is_na.groupby(gap_groups).transform("sum")
        # Only interpolate small gaps
        small_gap_mask = is_na & (gap_sizes <= max_gap)
        if small_gap_mask.any():
            interpolated = series.interpolate(method=method)
            df.loc[small_gap_mask, col] = interpolated.loc[small_gap_mask]

    return df
