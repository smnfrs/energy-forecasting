"""Day-boundary-aware time series cross-validation.

All splits enforce full-day boundaries:
- Train ends at hour 23
- Test starts at hour 0
- Test/train sizes rounded to whole days

Holdout is carved out BEFORE CV — CV never sees holdout data.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class TimeSeriesSplitter:
    """Day-boundary-aware time series CV splitter.

    Compatible with sklearn's cross-validation API (has a ``split`` method
    returning train/test index arrays).

    Parameters
    ----------
    n_splits : int
        Number of CV folds. Caller must pass explicitly (SEARCH_CV_FOLDS
        or VALIDATION_CV_FOLDS from config/modeling.py).
    test_days : int or None
        Size of each test fold in days. If None, the available pool is
        divided evenly across folds.
    mode : str
        "expanding" (train grows from the start) or "sliding" (fixed-size
        train window).
    gap_days : int
        Gap between train and test in days (to avoid leakage from lagged
        features). Default 0.
    step_days : int or None
        How many days to step between folds. None = non-overlapping
        (step = test_days).
    """

    def __init__(
        self,
        n_splits: int,
        test_days: int | None = None,
        mode: str = "expanding",
        gap_days: int = 0,
        step_days: int | None = None,
    ):
        if mode not in ("expanding", "sliding"):
            raise ValueError(f"mode must be 'expanding' or 'sliding', got '{mode}'")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.n_splits = n_splits
        self.test_days = test_days
        self.mode = mode
        self.gap_days = gap_days
        self.step_days = step_days

    def split(
        self,
        index: pd.DatetimeIndex,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index arrays.

        Parameters
        ----------
        index : pd.DatetimeIndex
            The full index (after holdout removal). Must be hourly.

        Yields
        ------
        train_idx, test_idx : tuple of np.ndarray
            Integer position indices into `index`.
        """
        dates = index.normalize().unique().sort_values()
        n_days = len(dates)

        # Determine test fold size
        if self.test_days is not None:
            test_days = self.test_days
        else:
            # Divide pool evenly: need n_splits test folds + at least 1 day of train
            test_days = max(1, (n_days - 1) // self.n_splits)

        step_days = self.step_days if self.step_days is not None else test_days

        # Build folds from the newest data backward
        folds = []
        for i in range(self.n_splits):
            test_end_idx = n_days - i * step_days
            test_start_idx = test_end_idx - test_days
            train_end_idx = test_start_idx - self.gap_days

            if test_start_idx < 0 or train_end_idx < 1:
                break

            test_start_date = dates[test_start_idx]
            test_end_date = dates[test_end_idx - 1]  # inclusive

            train_end_date = dates[train_end_idx - 1]  # inclusive

            if self.mode == "expanding":
                train_start_date = dates[0]
            else:
                # sliding: fixed train size = total available minus what's after
                train_size_days = train_end_idx
                min_train = max(test_days, 30)  # at least 30 days or test_days
                if train_size_days < min_train:
                    break
                train_start_date = dates[0]
                # In sliding mode, use a fixed window from the end of available train
                sliding_train_days = n_days - self.n_splits * step_days - self.gap_days
                if sliding_train_days < min_train:
                    sliding_train_days = min_train
                actual_start_idx = max(0, train_end_idx - sliding_train_days)
                train_start_date = dates[actual_start_idx]

            # Convert date boundaries to hourly index positions
            # Train: [train_start_date 00:00, train_end_date 23:xx]
            # Test: [test_start_date 00:00, test_end_date 23:xx]
            train_mask = (index >= train_start_date) & (
                index <= train_end_date + pd.Timedelta(hours=23)
            )
            test_mask = (index >= test_start_date) & (
                index <= test_end_date + pd.Timedelta(hours=23)
            )

            train_positions = np.where(train_mask)[0]
            test_positions = np.where(test_mask)[0]

            if len(train_positions) == 0 or len(test_positions) == 0:
                break

            folds.append((train_positions, test_positions))

        # Reverse so fold 1 has smallest train (oldest test), fold N has largest
        yield from reversed(folds)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """sklearn compatibility."""
        return self.n_splits


def carve_holdout(
    index: pd.DatetimeIndex,
    holdout_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split index into train+CV pool and holdout.

    Holdout is the last `holdout_days` full days. Boundary is at day start
    (hour 0). Returns integer position indices.
    """
    dates = index.normalize().unique().sort_values()
    if holdout_days >= len(dates):
        raise ValueError(f"holdout_days ({holdout_days}) >= total days ({len(dates)})")

    holdout_start = dates[-holdout_days]
    pool_mask = index < holdout_start
    holdout_mask = index >= holdout_start

    return np.where(pool_mask)[0], np.where(holdout_mask)[0]
