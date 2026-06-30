"""Forecasting functions for gen/load models.

Direct prediction is the default. Recursive forecasting (with lag updates)
is optional — activated only when the model uses lagged target features.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.modeling.intervals import predict_with_intervals


def find_target_lag_columns(
    columns: Iterable[str],
    target_name: str,
) -> list[str]:
    """Return columns matching the `{target_name}_h{N}` pattern.

    Target-lag columns are autoregressive features that require recursive
    forecasting at inference time. They are distinguished from TSO-level
    lag features (e.g. ``gen_wind_on_h24``) by the target-name prefix.
    """
    prefix = f"{target_name}_h"
    matches: list[tuple[int, str]] = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            matches.append((int(suffix), col))
    matches.sort()
    return [col for _, col in matches]


def forecast_direct(model, X_test: pd.DataFrame) -> pd.DataFrame:
    """Direct prediction — no lag updates. Default mode.

    Works with both MAPIE-wrapped and plain sklearn models.
    Returns DataFrame with columns: fitted, lower, upper.
    If model is not MAPIE-wrapped, lower/upper are NaN.
    """
    has_intervals = hasattr(model, "predict_interval")

    if has_intervals:
        y_pred, y_lower, y_upper = predict_with_intervals(model, X_test)
    else:
        y_pred = np.asarray(model.predict(X_test))
        y_lower = np.full_like(y_pred, np.nan)
        y_upper = np.full_like(y_pred, np.nan)

    return pd.DataFrame(
        {"fitted": y_pred, "lower": y_lower, "upper": y_upper},
        index=X_test.index,
    )


#: Matches EMA's `FORECAST_HORIZON = 168` (1 week, hourly). Recursive lag
#: forecasts beyond this horizon compound errors meaninglessly and are also
#: far slower than any practical use requires. EMA's CV calls
#: `forecast_window` once per 168h test fold.
DEFAULT_FORECAST_HORIZON = 168


def forecast_with_lags_windowed(
    model,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    lag_columns: list[str],
    window_size: int = DEFAULT_FORECAST_HORIZON,
    sample_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run recursive forecasts over non-overlapping windows of X_test.

    Each window is a fresh ``forecast_with_lags`` call: its lag seeds come
    from the pre-filled X_test lag columns (which reflect actuals via
    ``y.shift`` on the full raw series), and its predictions overwrite
    subsequent rows **within that window only**. Errors do not compound
    across window boundaries — matching EMA's 168h forecast horizon.

    Parameters
    ----------
    window_size : int
        Length of each recursive forecast window. 168 = 1 week.
    sample_windows : int or None
        If ``None``, evaluate every non-overlapping window covering X_test
        (used on holdout). Otherwise, evaluate this many non-overlapping
        windows evenly spaced across X_test (used in CV to keep search
        tractable on multi-year test folds).

    Returns
    -------
    y_pred : np.ndarray
        Length ``len(X_test)``; rows outside evaluated windows are NaN.
    eval_mask : np.ndarray
        Boolean array marking which rows were evaluated.
    """
    if not lag_columns:
        raise ValueError("forecast_with_lags_windowed requires lag_columns")

    n = len(X_test)
    max_windows = max(1, n // window_size)

    if sample_windows is None or sample_windows >= max_windows:
        window_indices = list(range(max_windows))
    elif sample_windows == 1:
        # Single window: the latest one (most recent data in the fold)
        window_indices = [max_windows - 1]
    else:
        # Evenly spaced, both endpoints included
        window_indices = list(np.linspace(0, max_windows - 1, sample_windows, dtype=int))

    y_pred = np.full(n, np.nan)
    eval_mask = np.zeros(n, dtype=bool)

    for wi in window_indices:
        start = wi * window_size
        end = min(start + window_size, n)
        if end - start <= 0:
            continue
        X_window = X_test.iloc[start:end]
        result = forecast_with_lags(model, X_window, y_train, lag_columns)
        y_pred[start:end] = result["fitted"].values
        eval_mask[start:end] = True

    return y_pred, eval_mask


def forecast_with_lags(
    model,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    lag_columns: list[str],
    target_name: str = "",
) -> pd.DataFrame:
    """Recursive one-step-ahead forecasting with lag feature updates.

    For each timestep, predicts the target, then overwrites the lag columns
    of subsequent rows with the prediction. X_test is assumed to have lag
    columns pre-filled from ``y.shift(lag)`` on the full series — so row 0's
    lag values correctly come from ``y_train``'s tail. Each subsequent row's
    lag values get overwritten in place as predictions become available.

    This matches EMA's ``forecast_window`` pattern: no train/inference
    mismatch because both CV and inference see lag features populated from
    the model's own (possibly-error-compounded) predictions rather than from
    leaked actuals.

    Parameters
    ----------
    model : fitted model (sklearn or MAPIE-wrapped)
    X_test : DataFrame with lag columns pre-filled from ``y.shift(lag)`` on
        the full series. Row 0's lag values must already reflect ``y_train``'s
        tail (the standard outcome of building features before splitting).
    y_train : training target (retained for future extensibility; not used
        by the current implementation because X_test's pre-fill already
        carries the seed values).
    lag_columns : list of column names containing lagged target values.
        Must be non-empty.
    target_name : name of the target (unused; kept for caller clarity).
    """
    if not lag_columns:
        raise ValueError("forecast_with_lags requires at least one lag column")

    del y_train, target_name  # retained for API stability
    has_intervals = hasattr(model, "predict_interval")

    X_work = X_test.copy()
    predictions = np.zeros(len(X_test))
    lowers = np.full(len(X_test), np.nan)
    uppers = np.full(len(X_test), np.nan)

    # Pre-compute column positions and lag offsets once
    col_positions = [(_extract_lag(col), X_work.columns.get_loc(col)) for col in lag_columns]

    for i in range(len(X_test)):
        row = X_work.iloc[[i]]

        if has_intervals:
            y_pred, y_lo, y_up = predict_with_intervals(model, row)
            predictions[i] = y_pred[0]
            lowers[i] = y_lo[0]
            uppers[i] = y_up[0]
        else:
            predictions[i] = model.predict(row)[0]

        # Overwrite lag columns of future rows with the fresh prediction.
        # At row i+lag, the lag-hours-ago value is the prediction we just
        # made for row i.
        for lag, col_pos in col_positions:
            target_row = i + lag
            if target_row < len(X_test):
                X_work.iat[target_row, col_pos] = predictions[i]

    logger.debug(f"Recursive forecast: {len(X_test)} steps")

    return pd.DataFrame(
        {"fitted": predictions, "lower": lowers, "upper": uppers},
        index=X_test.index,
    )


def _extract_lag(column_name: str) -> int:
    """Extract lag value from a column name like 'target_h24' → 24."""
    parts = column_name.split("_h")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    raise ValueError(f"Cannot extract lag from column name: {column_name}")
