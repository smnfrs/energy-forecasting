"""Prediction interval wrappers.

Base model intervals via MAPIE 1.3 CrossConformalRegressor.
Ensemble intervals via post-hoc conformal calibration (method-agnostic).
"""

from __future__ import annotations

import numpy as np
from mapie.regression import CrossConformalRegressor
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from energy_forecasting.config.modeling import PI_CONFIDENCE_LEVEL, PI_CV_FOLDS


def wrap_with_intervals(
    estimator: BaseEstimator,
    confidence_level: float = PI_CONFIDENCE_LEVEL,
    cv: int = PI_CV_FOLDS,
    n_jobs: int | None = None,
) -> CrossConformalRegressor:
    """Wrap an sklearn estimator with CrossConformalRegressor."""
    return CrossConformalRegressor(
        estimator=estimator,
        confidence_level=confidence_level,
        cv=cv,
        n_jobs=n_jobs,
    )


def predict_with_intervals(
    model: CrossConformalRegressor,
    X,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict with intervals from a fitted CrossConformalRegressor.

    Returns (point_pred, lower, upper).
    """
    y_pred, intervals = model.predict_interval(X)
    # intervals shape: (n_samples, 2, 1) for single confidence level
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    return np.asarray(y_pred), np.asarray(lower), np.asarray(upper)


# ── Ensemble intervals (post-hoc conformal) ───────────────────────


def calibrate_ensemble_intervals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    confidence_level: float = PI_CONFIDENCE_LEVEL,
) -> float:
    """Compute conformal quantile from holdout residuals.

    The quantile q is chosen so that coverage(y_true, y_pred ± q) >= confidence_level.
    This is method-agnostic — works for any ensemble.

    Returns the conformal quantile (scalar).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = np.abs(y_true - y_pred)
    n = len(residuals)

    # Conformal quantile: ceil((n+1) * confidence_level) / n percentile
    quantile_level = np.ceil((n + 1) * confidence_level) / n
    quantile_level = min(quantile_level, 1.0)

    return float(np.quantile(residuals, quantile_level))


def predict_ensemble_intervals(
    y_pred: ArrayLike,
    conformal_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conformal quantile to get ensemble prediction intervals.

    Returns (lower, upper) = (y_pred - q, y_pred + q).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    lower = y_pred - conformal_quantile
    upper = y_pred + conformal_quantile
    return lower, upper
