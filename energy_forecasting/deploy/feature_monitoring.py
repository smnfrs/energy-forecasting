"""Feature availability monitoring for deployment inference runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def now_utc_iso() -> str:
    """Return an ISO UTC timestamp with a stable dashboard-friendly shape."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values to JSON-compatible Python values."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def summarize_matrix(
    matrix: pd.DataFrame,
    *,
    timestamps: pd.DatetimeIndex | None = None,
    max_columns: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Summarize completeness of a model feature matrix.

    The summary deliberately records counts and examples, not full matrices, so
    the deployed artifact stays small while still showing which columns failed.
    """
    timestamps = timestamps if timestamps is not None else matrix.index
    na = matrix.isna()
    rows_missing = na.sum(axis=1)
    cols_missing = na.sum(axis=0)
    bad_cols = cols_missing[cols_missing > 0].sort_values(ascending=False)
    bad_rows = rows_missing[rows_missing > 0]

    column_items = bad_cols.items() if max_columns is None else bad_cols.head(max_columns).items()
    row_index = bad_rows.index if max_rows is None else bad_rows.head(max_rows).index
    sample_limit = len(matrix.columns) if max_columns is None else max_columns

    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "complete_rows": int((rows_missing == 0).sum()),
        "nan_cells": int(na.to_numpy().sum()),
        "columns_with_nan": [
            {"name": str(name), "nan_count": int(count)}
            for name, count in column_items
        ],
        "rows_with_nan": [
            {
                "timestamp": _json_value(ts),
                "nan_count": int(rows_missing.loc[ts]),
                "sample_columns": [str(c) for c in matrix.columns[na.loc[ts]].tolist()[:sample_limit]],
            }
            for ts in row_index
        ],
        "first_timestamp": _json_value(timestamps[0]) if len(timestamps) else None,
        "last_timestamp": _json_value(timestamps[-1]) if len(timestamps) else None,
    }


def dataframe_records(matrix: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a feature matrix to timestamped JSON records for full audits."""
    records: list[dict[str, Any]] = []
    for ts, row in matrix.iterrows():
        item: dict[str, Any] = {"timestamp": _json_value(ts)}
        for col, value in row.items():
            item[str(col)] = _json_value(value)
        records.append(item)
    return records


def summarize_source_availability(
    original_df: pd.DataFrame,
    forecast_idx: pd.DatetimeIndex,
    *,
    exclude_columns: set[str] | None = None,
    max_stale_columns: int | None = None,
) -> dict[str, Any]:
    """Summarize raw input availability before deployment forward-fill.

    This is the key diagnostic for deployment degradation: if a forecast uses a
    value carried forward from hours or days before the delivery window, the
    audit records that staleness explicitly.
    """
    exclude_columns = exclude_columns or set()
    start = forecast_idx[0] if len(forecast_idx) else None
    end = forecast_idx[-1] if len(forecast_idx) else None
    if start is None or end is None:
        return {"columns": 0, "stale_columns": [], "forecast_window": None}

    source_cols = [c for c in original_df.columns if c not in exclude_columns]
    stale: list[dict[str, Any]] = []
    present_in_window = 0

    for col in source_cols:
        series = original_df[col]
        window = series.reindex(forecast_idx)
        if window.notna().any():
            present_in_window += 1
            continue
        valid = series.loc[:end].dropna()
        last_seen = valid.index[-1] if not valid.empty else None
        hours_stale = None
        if last_seen is not None:
            hours_stale = (start - last_seen) / pd.Timedelta(hours=1)
        stale.append(
            {
                "name": str(col),
                "last_observed": _json_value(last_seen),
                "hours_stale_at_start": _json_value(hours_stale),
            }
        )

    stale.sort(
        key=lambda item: (
            item["hours_stale_at_start"] is None,
            -(item["hours_stale_at_start"] or -10**9),
            item["name"],
        )
    )
    return {
        "columns": len(source_cols),
        "columns_present_in_forecast_window": present_in_window,
        "columns_filled_from_history": len(stale),
        "forecast_window": {
            "start": _json_value(start),
            "end": _json_value(end),
        },
        "stale_columns": stale if max_stale_columns is None else stale[:max_stale_columns],
    }


def model_expected_features(model: Any) -> list[str]:
    """Best-effort extraction of feature names expected by a fitted model."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(c) for c in list(names)]
    # Pipelines usually expose names on the final estimator.
    steps = getattr(model, "steps", None)
    if steps:
        estimator = steps[-1][1]
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            return [str(c) for c in list(names)]
    return []
