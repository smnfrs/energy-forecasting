"""Metric calculations for model evaluation.

All metrics operate on numpy arrays / pandas Series and return plain dicts
suitable for MLflow logging.
"""

import numpy as np
from numpy.typing import ArrayLike

from energy_forecasting.config.modeling import PEAK_HOURS


def calculate_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_baseline: ArrayLike | None = None,
) -> dict[str, float]:
    """Core regression metrics.

    Returns: RMSE, MAE, ME (bias), R², MAPE, sMAPE.
    If y_baseline is provided, adds MAE_skill and RMSE_skill
    (1 - model_error/baseline_error; >0 means model beats baseline).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    me = float(np.mean(residuals))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE — skip zeros in y_true to avoid division by zero
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.any():
        mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100)
    else:
        mape = float("inf")

    # sMAPE
    denom = np.abs(y_true) + np.abs(y_pred)
    nonzero_denom = denom > 1e-8
    if nonzero_denom.any():
        smape = float(np.mean(2.0 * np.abs(residuals[nonzero_denom]) / denom[nonzero_denom]) * 100)
    else:
        smape = float("inf")

    result = {
        "mae": mae,
        "rmse": rmse,
        "me": me,
        "r2": r2,
        "mape": mape,
        "smape": smape,
    }

    if y_baseline is not None:
        y_baseline = np.asarray(y_baseline, dtype=float)
        baseline_residuals = y_true - y_baseline
        baseline_mae = float(np.mean(np.abs(baseline_residuals)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_residuals**2)))
        result["mae_skill"] = float(1.0 - mae / baseline_mae) if baseline_mae > 0 else 0.0
        result["rmse_skill"] = float(1.0 - rmse / baseline_rmse) if baseline_rmse > 0 else 0.0

    return result


def calculate_pi_metrics(
    y_true: ArrayLike,
    y_lower: ArrayLike,
    y_upper: ArrayLike,
) -> dict[str, float]:
    """Prediction interval metrics.

    Returns: coverage (fraction of y_true within [lower, upper]),
    mean_width, median_width.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)

    covered = (y_true >= y_lower) & (y_true <= y_upper)
    widths = y_upper - y_lower

    return {
        "pi_coverage": float(np.mean(covered)),
        "pi_mean_width": float(np.mean(widths)),
        "pi_median_width": float(np.median(widths)),
    }


def calculate_peak_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    index=None,
    peak_hours: list[int] | None = None,
) -> dict[str, float]:
    """Metrics restricted to peak hours (default 8-19).

    Requires a DatetimeIndex (passed as `index`) to identify hours.
    Returns peak_mae, peak_rmse, peak_me.
    """
    if peak_hours is None:
        peak_hours = PEAK_HOURS
    if index is None:
        raise ValueError("index (DatetimeIndex) required for peak metrics")

    mask = np.isin(index.hour, peak_hours)
    if not mask.any():
        return {"peak_mae": float("nan"), "peak_rmse": float("nan"), "peak_me": float("nan")}

    y_true_peak = np.asarray(y_true, dtype=float)[mask]
    y_pred_peak = np.asarray(y_pred, dtype=float)[mask]
    residuals = y_true_peak - y_pred_peak

    return {
        "peak_mae": float(np.mean(np.abs(residuals))),
        "peak_rmse": float(np.sqrt(np.mean(residuals**2))),
        "peak_me": float(np.mean(residuals)),
    }
