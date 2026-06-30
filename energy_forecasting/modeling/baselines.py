"""Naive and statistical baselines for model evaluation.

Price baselines use 24h/168h lags. Gen/load baselines use 7-day persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_lag(y: pd.Series, lag: int = 24) -> pd.Series:
    """Same hour yesterday (lag=24) or other lag. Returns shifted series."""
    return y.shift(lag)


def naive_weekly(y: pd.Series) -> pd.Series:
    """Same hour, same day of week, previous week. Lag = 168h."""
    return y.shift(168)


def naive_persistence_7d(y: pd.Series) -> pd.Series:
    """Same hour, same day-of-week, previous week. Alias for 7-day gen/load baseline."""
    return y.shift(168)


def naive_seasonal_7d(y: pd.Series, n_weeks: int = 4) -> pd.Series:
    """Same hour, same day-of-week, average of last n_weeks weeks."""
    shifts = [y.shift(168 * w) for w in range(1, n_weeks + 1)]
    stacked = pd.concat(shifts, axis=1)
    return stacked.mean(axis=1)


def climatological_baseline(
    y: pd.Series,
    window_days: int = 90,
) -> pd.Series:
    """Hour-of-day × day-of-week mean over trailing window.

    For each timestamp, computes the mean of all observations from the
    preceding `window_days` that share the same hour and day-of-week.
    """
    result = pd.Series(np.nan, index=y.index)
    hours = y.index.hour
    dows = y.index.dayofweek
    window = pd.Timedelta(days=window_days)

    # Group by (hour, dow) for efficiency
    for (hour, dow), group_idx in y.groupby([hours, dows]).groups.items():
        for idx in group_idx:
            ts = y.index[idx]
            lookback_start = ts - window
            mask = (
                (y.index >= lookback_start)
                & (y.index < ts)
                & (hours == hour)
                & (dows == dow)
            )
            vals = y[mask]
            if len(vals) > 0:
                result.iloc[idx] = vals.mean()

    return result
