"""Training loop with MLflow integration.

Core function: train_model() — loads dataset, builds pipeline, runs CV,
fits MAPIE intervals, evaluates on holdout, logs everything to MLflow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from energy_forecasting.config.modeling import HOLDOUT_DAYS, PI_CONFIDENCE_LEVEL, PI_CV_FOLDS
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import load_dataset, log_dataset_to_run
from energy_forecasting.modeling.forecasting import (
    DEFAULT_FORECAST_HORIZON,
    forecast_with_lags_windowed,
)
from energy_forecasting.modeling.intervals import (
    predict_with_intervals,
    wrap_with_intervals,
)
from energy_forecasting.modeling.metrics import (
    calculate_metrics,
    calculate_peak_metrics,
    calculate_pi_metrics,
)
from energy_forecasting.modeling.mlflow_utils import TrackedRun

# Recursive CV evaluation uses `sample_windows=None` (full coverage over
# each test fold). With EMA-matching sliding-window CV (`test_days=7`),
# each test fold is already exactly one 168h window, so "full coverage"
# means a single recursive forecast call per fold — identical to EMA's
# `forecast_window` behaviour.

# ── Scalers ───────────────────────────────────────────────────────

_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


# ── Target transforms ────────────────────────────────────────────


class _LogShiftTransformer(BaseEstimator, TransformerMixin):
    """log1p(y + shift) transform where shift = |min(y)| + 1 if y has negatives.

    Inherits from BaseEstimator for get_params/set_params (needed by clone()).
    """

    def __init__(self):
        self.shift_ = 0.0

    def fit(self, y, **params):
        y = np.asarray(y)
        if y.min() <= 0:
            self.shift_ = float(np.abs(y.min()) + 1.0)
        else:
            self.shift_ = 0.0
        return self

    def transform(self, y):
        return np.log1p(np.asarray(y) + self.shift_)

    def inverse_transform(self, y):
        return np.expm1(np.asarray(y)) - self.shift_


def _make_target_regressor(model, target_transform: str):
    """Wrap model in TransformedTargetRegressor if needed."""
    if target_transform == "none":
        return model
    if target_transform == "log_shift":
        transformer = _LogShiftTransformer()
    elif target_transform == "yeo_johnson":
        transformer = PowerTransformer(method="yeo-johnson")
    else:
        raise ValueError(f"Unknown target_transform: {target_transform}")
    return TransformedTargetRegressor(regressor=model, transformer=transformer)


# ── Pipeline builder ──────────────────────────────────────────────


def build_pipeline(
    model,
    scaler: str = "standard",
    target_transform: str = "none",
) -> Pipeline:
    """Build sklearn Pipeline: scaler → TransformedTargetRegressor(model).

    Parameters
    ----------
    model : sklearn-compatible regressor
    scaler : "standard", "robust", or "none"
    target_transform : "none", "log_shift", or "yeo_johnson"
    """
    steps = []

    scaler_cls = _SCALERS.get(scaler)
    if scaler_cls is not None:
        steps.append(("scaler", scaler_cls()))

    wrapped_model = _make_target_regressor(model, target_transform)
    steps.append(("model", wrapped_model))

    return Pipeline(steps)


# ── Sample weights ────────────────────────────────────────────────


def compute_sample_weights(day_index: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential decay weights. w = exp(ln(2)/half_life × (t - t_max)).

    Parameters
    ----------
    day_index : pd.Series
        Numeric day index (from temporal features or computed from DatetimeIndex).
    half_life_days : float
        At half_life days from the most recent observation, weight = 0.5.
    """
    days = np.asarray(day_index, dtype=float)
    t_max = days.max()
    return np.exp(np.log(2) / half_life_days * (days - t_max))


# ── Core training function ────────────────────────────────────────


def train_model(
    dataset_path: Path,
    model,
    experiment: str,
    tags: dict,
    scaler: str = "standard",
    target_transform: str = "none",
    weight_half_life: float | None = None,
    cv: TimeSeriesSplitter | None = None,
    holdout_days: int = HOLDOUT_DAYS,
    confidence_level: float = PI_CONFIDENCE_LEVEL,
    pi_cv_folds: int = PI_CV_FOLDS,
    collect_oof: bool = False,
    target_lag_columns: list[str] | None = None,
    test_dataset_path: Path | None = None,
) -> str:
    """Train a model with full MLflow tracking.

    Steps:
    1. Load dataset from Parquet.
    2. Open TrackedRun — validates tags + experiment BEFORE training.
    3. Carve holdout (last holdout_days).
    4. Build pipeline (scaler → model) for CV evaluation.
    5. CV evaluation on train portion (if cv provided).
    6. Pre-scale features, wrap model with MAPIE, fit_conformalize.
       MAPIE doesn't support sklearn Pipeline routing for sample_weight,
       so we pre-scale X and pass a bare TransformedTargetRegressor.
    7. Holdout evaluation with prediction intervals.
    8. Log everything to MLflow.
    9. Return run_id.

    If collect_oof=True (requires cv), out-of-fold predictions from each CV
    test fold are concatenated and logged as an MLflow artifact. These are
    used by stacking ensembles (ensemble_gen_load, price ensemble).

    If ``target_lag_columns`` is non-empty, the dataset contains
    autoregressive target-lag features (e.g. ``wind_onshore_h1`` through
    ``wind_onshore_h12``). In that case CV and holdout evaluation use
    recursive forecasting (feeding each hour's prediction into subsequent
    rows' lag columns) to match inference conditions. MAPIE is bypassed for
    such runs — their prediction intervals are reported as NaN, mirroring
    EMA's behaviour which similarly skips conformal wrapping for the
    recursive path.
    """
    X, y = load_dataset(dataset_path)
    dataset_name = dataset_path.stem
    use_recursive = bool(target_lag_columns)

    # Optional alternate feature matrix used for CV test folds and holdout
    # prediction. Same index/columns as ``X`` but built from a different
    # weather source (gen/load: ``hist_forecast`` vs ``history``). Training
    # rows still come from ``X``; only test/holdout features are swapped.
    # This produces backtest-honest OOF predictions equivalent to EMA's
    # ``mode="backtest"`` in ``generate_historical_forecasts.py``.
    if test_dataset_path is not None:
        X_test_alt, _ = load_dataset(test_dataset_path)
        if not X_test_alt.index.equals(X.index):
            raise ValueError(
                f"test_dataset_path index does not match dataset_path index "
                f"(diff: {len(X.index.symmetric_difference(X_test_alt.index))} rows)"
            )
        if list(X_test_alt.columns) != list(X.columns):
            raise ValueError("test_dataset_path columns do not match dataset_path columns")
    else:
        X_test_alt = None

    # Carve holdout
    pool_idx, holdout_idx = carve_holdout(X.index, holdout_days)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_holdout, y_holdout = X.iloc[holdout_idx], y.iloc[holdout_idx]
    if X_test_alt is not None:
        X_pool_test = X_test_alt.iloc[pool_idx]
        X_holdout_test = X_test_alt.iloc[holdout_idx]
    else:
        X_pool_test = X_pool
        X_holdout_test = X_holdout

    pipeline = build_pipeline(model, scaler=scaler, target_transform=target_transform)

    # Compute sample weights for the full pool
    pool_weights = None
    if weight_half_life is not None:
        days = (X_pool.index - X_pool.index[0]).total_seconds() / 86400.0
        pool_weights = compute_sample_weights(pd.Series(days), weight_half_life)

    with TrackedRun(experiment, dataset_name=dataset_name, **tags) as run:
        run_id = run.info.run_id

        # Log dataset provenance
        log_dataset_to_run(X, dataset_path)

        # Auto-tags
        mlflow.set_tag("model_class", type(model).__name__)
        mlflow.set_tag("n_features", str(X.shape[1]))
        mlflow.set_tag("n_train_rows", str(len(X_pool)))
        mlflow.set_tag("holdout_days", str(holdout_days))
        if cv is not None:
            mlflow.set_tag("cv_mode", cv.mode)
            mlflow.set_tag("cv_folds", str(cv.n_splits))

        # Log preprocessing params
        mlflow.log_params(
            {
                "scaler": scaler,
                "target_transform": target_transform,
                "weight_half_life": str(weight_half_life),
                "holdout_days": holdout_days,
            }
        )

        # ── CV evaluation (uses Pipeline for scaler routing) ───
        # When autoregressive lag columns are present, each fold predicts
        # recursively — feeding fresh predictions into the lag columns of
        # subsequent rows — so CV metrics reflect inference-time behaviour
        # rather than the leaky "lag_1 == true y[t-1]" shortcut.
        oof_parts: list[pd.DataFrame] = []
        if cv is not None:
            cv_metrics_list = []
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_pool.index)):
                fold_pipeline = clone(pipeline)
                X_train_fold = X_pool.iloc[train_idx]
                y_train_fold = y_pool.iloc[train_idx]
                # Test rows come from the alt frame when provided — gives
                # backtest-honest OOF predictions when X_test_alt is built
                # from forecast weather.
                X_test_fold = X_pool_test.iloc[test_idx]
                y_test_fold = y_pool.iloc[test_idx]

                fold_fit_params = {}
                if pool_weights is not None:
                    fold_fit_params["model__sample_weight"] = pool_weights[train_idx]

                fold_pipeline.fit(X_train_fold, y_train_fold, **fold_fit_params)
                if use_recursive:
                    y_pred_full, eval_mask = forecast_with_lags_windowed(
                        fold_pipeline,
                        X_test_fold,
                        y_train_fold,
                        target_lag_columns,
                        window_size=DEFAULT_FORECAST_HORIZON,
                        sample_windows=None,
                    )
                    y_true_eval = y_test_fold.values[eval_mask]
                    y_pred_eval = y_pred_full[eval_mask]
                    fold_metrics = calculate_metrics(y_true_eval, y_pred_eval)
                    cv_metrics_list.append(fold_metrics)

                    if collect_oof:
                        oof_parts.append(
                            pd.DataFrame(
                                {"y_true": y_true_eval, "y_pred": y_pred_eval},
                                index=X_test_fold.index[eval_mask],
                            )
                        )
                else:
                    y_pred_fold = fold_pipeline.predict(X_test_fold)
                    fold_metrics = calculate_metrics(y_test_fold, y_pred_fold)
                    cv_metrics_list.append(fold_metrics)

                    if collect_oof:
                        oof_parts.append(
                            pd.DataFrame(
                                {"y_true": y_test_fold.values, "y_pred": y_pred_fold},
                                index=X_test_fold.index,
                            )
                        )

            cv_means = {
                f"cv_{k}": float(np.mean([m[k] for m in cv_metrics_list]))
                for k in cv_metrics_list[0]
            }
            mlflow.log_metrics(cv_means)
            logger.info(f"CV metrics: MAE={cv_means['cv_mae']:.2f}")

        # ── Final training ────────────────────────────────────
        # Three paths:
        #
        # RECURSIVE LAGS: skip MAPIE entirely (MAPIE's internal CV calls
        #   `.predict()` directly and has no hook for recursive lag updates,
        #   so its conformity scores would be computed against leaky direct
        #   predictions). Fit the pipeline on the full pool and run
        #   `forecast_with_lags` over the holdout. PI metrics are reported
        #   as NaN — matching EMA which also bypasses conformal wrapping
        #   on the recursive path.
        #
        # WITHOUT weights: pass the full Pipeline to MAPIE directly.
        #   Trees get unscaled X (scaler="none"), linear models get the
        #   Pipeline with scaler built in. No workaround needed.
        #
        # WITH weights: MAPIE's CrossConformalRegressor doesn't support
        #   sklearn Pipeline routing for fit_params (model__sample_weight).
        #   Workaround: pre-scale X externally, pass a bare estimator (or
        #   TransformedTargetRegressor) to MAPIE with fit_params={"sample_weight": w}.
        #   The scaler is saved as a separate MLflow artifact for inference.

        mapie_model = None
        holdout_eval_mask = None
        if use_recursive:
            logger.warning(
                f"Target lag columns present ({len(target_lag_columns)}); "
                f"bypassing MAPIE and using windowed recursive forecasting "
                f"on holdout. Prediction intervals will be NaN.",
            )
            fitted_pipeline = clone(pipeline)
            fit_kwargs: dict = {}
            if pool_weights is not None:
                fit_kwargs["model__sample_weight"] = pool_weights
            fitted_pipeline.fit(X_pool, y_pool, **fit_kwargs)
            # Full coverage of the holdout via non-overlapping 168h windows.
            # Each window re-seeds its lag state from actuals in X_holdout
            # (pre-filled via y.shift on the full series), so errors compound
            # only within a 1-week horizon — matching EMA's evaluation.
            y_pred, holdout_eval_mask = forecast_with_lags_windowed(
                fitted_pipeline,
                X_holdout_test,
                y_pool,
                target_lag_columns,
                window_size=DEFAULT_FORECAST_HORIZON,
                sample_windows=None,
            )
            y_lower = np.full_like(y_pred, np.nan)
            y_upper = np.full_like(y_pred, np.nan)
            scaler_obj = None
        elif pool_weights is None:
            # Simple path — Pipeline goes directly into MAPIE
            mapie_model = wrap_with_intervals(
                clone(pipeline),
                confidence_level=confidence_level,
                cv=pi_cv_folds,
            )
            mapie_model.fit_conformalize(X_pool, y_pool)
            # Holdout uses alt features when provided — model trained on
            # actual weather, evaluated on hist_forecast weather. Mirrors
            # production inference where features come from forecast weather.
            y_pred, y_lower, y_upper = predict_with_intervals(
                mapie_model,
                X_holdout_test,
            )
            scaler_obj = None  # scaler is inside the Pipeline
        else:
            # Workaround path — pre-scale, bare estimator, fit_params dict
            scaler_obj = _fit_scaler(scaler, X_pool)
            X_pool_scaled = _apply_scaler(scaler_obj, X_pool)
            X_holdout_test_scaled = _apply_scaler(scaler_obj, X_holdout_test)

            mapie_estimator = _make_target_regressor(clone(model), target_transform)
            mapie_model = wrap_with_intervals(
                mapie_estimator,
                confidence_level=confidence_level,
                cv=pi_cv_folds,
            )
            mapie_model.fit_conformalize(
                X_pool_scaled,
                y_pool,
                fit_params={"sample_weight": pool_weights},
            )
            y_pred, y_lower, y_upper = predict_with_intervals(
                mapie_model,
                X_holdout_test_scaled,
            )

        # ── Holdout evaluation ─────────────────────────────────
        if holdout_eval_mask is not None:
            y_holdout_arr = np.asarray(y_holdout)[holdout_eval_mask]
            y_pred_arr = y_pred[holdout_eval_mask]
            eval_index = X_holdout.index[holdout_eval_mask]
        else:
            y_holdout_arr = np.asarray(y_holdout)
            y_pred_arr = y_pred
            eval_index = X_holdout.index

        holdout_metrics = calculate_metrics(y_holdout_arr, y_pred_arr)
        peak_metrics = calculate_peak_metrics(
            y_holdout_arr,
            y_pred_arr,
            index=eval_index,
        )

        mlflow.log_metrics(holdout_metrics)
        mlflow.log_metrics(peak_metrics)

        if use_recursive:
            pi_metrics: dict[str, float] = {}
        else:
            pi_metrics = calculate_pi_metrics(
                np.asarray(y_holdout),
                y_lower,
                y_upper,
            )
            mlflow.log_metrics(pi_metrics)

        # Log model artifact (+ separate scaler if workaround path was used).
        # Recursive-lag runs log the fitted pipeline directly since MAPIE
        # was bypassed.
        model_to_log = fitted_pipeline if use_recursive else mapie_model
        mlflow.sklearn.log_model(model_to_log, "model")
        if scaler_obj is not None:
            mlflow.sklearn.log_model(scaler_obj, "scaler")

        # Log holdout predictions as artifact. For recursive runs only the
        # rows actually evaluated are included — rows outside the sampled
        # windows have no honest prediction.
        if holdout_eval_mask is not None:
            holdout_preds = pd.DataFrame(
                {
                    "y_true": y_holdout_arr,
                    "y_pred": y_pred_arr,
                    "y_lower": y_lower[holdout_eval_mask],
                    "y_upper": y_upper[holdout_eval_mask],
                },
                index=eval_index,
            )
        else:
            holdout_preds = pd.DataFrame(
                {
                    "y_true": y_holdout.values,
                    "y_pred": y_pred,
                    "y_lower": y_lower,
                    "y_upper": y_upper,
                },
                index=X_holdout.index,
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "holdout_predictions.parquet"
            holdout_preds.to_parquet(path)
            mlflow.log_artifact(str(path), "predictions")

        # Log OOF predictions if collected
        if oof_parts:
            oof_df = pd.concat(oof_parts).sort_index()
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "oof_predictions.parquet"
                oof_df.to_parquet(path)
                mlflow.log_artifact(str(path), "predictions")

        pi_cov = pi_metrics.get("pi_coverage")
        pi_str = f"{pi_cov:.2%}" if pi_cov is not None and np.isfinite(pi_cov) else "n/a"
        logger.info(
            f"Run {run_id}: holdout MAE={holdout_metrics['mae']:.2f}, PI coverage={pi_str}"
        )

    return run_id


def _fit_scaler(scaler_name: str, X: pd.DataFrame):
    """Fit and return a scaler, or None if scaler_name is 'none'."""
    scaler_cls = _SCALERS.get(scaler_name)
    if scaler_cls is None:
        return None
    return scaler_cls().fit(X)


def _apply_scaler(scaler_obj, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted scaler, or return X as array if no scaler."""
    if scaler_obj is None:
        return np.asarray(X)
    return scaler_obj.transform(X)
