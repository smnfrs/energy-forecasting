"""Grid-search tuning for price models (§5c.3).

Two strategies:

* **Tree models** (LightGBM, XGBoost, CatBoost) — two-stage grid. Stage 1
  pins ``weight_half_life`` with a single representative config crossed
  against ``WEIGHT_HALF_LIVES``; stage 2 fixes the winning weight and
  walks ``PRICE_TREE_GRID``. Scaler and target transform are pinned to
  ``"none"`` because trees are invariant to monotonic feature transforms.

* **Linear models** (Ridge / Lasso) — exhaustive grid over
  ``LINEAR_PREPROCESSING_GRID × LINEAR_ALPHA_GRID``. Each trial is cheap
  (seconds) so we run them all. ElasticNet was dropped — Ridge + Lasso
  span the regularisation axis and ElasticNet's wider grid burned
  wall-clock for no measurable benefit.

Storage is Optuna SQLite via :class:`optuna.storages.RDBStorage`. Studies
resume on re-run — completed grid points are preserved across interruptions.

Every grid point also produces an MLflow run under
``price/model_training`` so the downstream candidate-selection step
(§5c.4) can rank them with the standard :class:`TrackedRun` machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from loguru import logger

from energy_forecasting.config.modeling import (
    HOLDOUT_DAYS,
    SEARCH_CV_FOLDS,
)
from energy_forecasting.config.search_spaces import (
    LINEAR_ALPHA_GRID,
    LINEAR_PREPROCESSING_GRID,
    PRICE_TREE_GRID,
    PRICE_TREE_WEIGHT_PROBE,
    WEIGHT_HALF_LIVES,
)
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import load_dataset
from energy_forecasting.modeling.metrics import calculate_metrics
from energy_forecasting.modeling.mlflow_utils import TrackedRun
from energy_forecasting.modeling.training import (
    build_pipeline,
    compute_sample_weights,
)

OPTUNA_DIR = Path("data/optuna")


# ── Model factory ────────────────────────────────────────────────


def _make_model(model_type: str, params: dict[str, Any]):
    """Instantiate a fresh model from ``model_type`` + hyperparams."""
    if model_type == "LGBMRegressor":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(verbose=-1, n_jobs=-1, **params)
    if model_type == "XGBRegressor":
        from xgboost import XGBRegressor

        return XGBRegressor(verbosity=0, n_jobs=-1, **params)
    if model_type == "CatBoostRegressor":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(thread_count=-1, allow_writing_files=False, **params)
    if model_type == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)
    if model_type == "Lasso":
        from sklearn.linear_model import Lasso

        return Lasso(max_iter=2_000, **params)
    raise ValueError(f"Unknown model_type {model_type!r}")


# ── Shared CV evaluator ─────────────────────────────────────────


def _evaluate_config(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    cv_splitter: TimeSeriesSplitter,
    model_type: str,
    model_params: dict[str, Any],
    scaler: str,
    target_transform: str,
    weight_half_life: float | None,
) -> dict[str, float]:
    """Return mean CV metrics over the splitter for one config."""
    weights = None
    if weight_half_life is not None:
        days = (X_pool.index - X_pool.index[0]).total_seconds() / 86400.0
        weights = compute_sample_weights(pd.Series(days), weight_half_life)

    fold_metrics: list[dict[str, float]] = []
    for train_idx, test_idx in cv_splitter.split(X_pool.index):
        model = _make_model(model_type, model_params)
        pipeline = build_pipeline(
            model, scaler=scaler, target_transform=target_transform,
        )
        fit_params: dict = {}
        if weights is not None:
            fit_params["model__sample_weight"] = weights[train_idx]

        pipeline.fit(X_pool.iloc[train_idx], y_pool.iloc[train_idx], **fit_params)
        preds = pipeline.predict(X_pool.iloc[test_idx])
        fold_metrics.append(calculate_metrics(y_pool.iloc[test_idx], preds))

    return {
        f"cv_{k}": float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0]
    }


# ── Study helpers ───────────────────────────────────────────────


def _study_storage(study_name: str) -> optuna.storages.RDBStorage:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(f"sqlite:///{OPTUNA_DIR / f'{study_name}.db'}")


def _grid_study(
    study_name: str,
    search_space: dict[str, list[Any]],
) -> optuna.Study:
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    return optuna.create_study(
        study_name=study_name,
        storage=_study_storage(study_name),
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )


def _log_trial_to_mlflow(
    *,
    experiment: str,
    feature_version: str,
    dataset_path: Path,
    model_type: str,
    tuning_step: str,
    model_params: dict[str, Any],
    scaler: str,
    target_transform: str,
    weight_half_life: float | None,
    cv_metrics: dict[str, float],
):
    """Persist one grid point as an MLflow run."""
    tags = {
        "stage": "model_training",
        "feature_version": feature_version,
        "holdout_days": str(HOLDOUT_DAYS),
        "cv_folds": str(SEARCH_CV_FOLDS),
        "cv_mode": "expanding",
        "target_transform": target_transform,
        "model_class": model_type,
        "tuning_strategy": "grid",
        "tuning_step": tuning_step,
    }
    with TrackedRun(experiment, dataset_name=dataset_path.stem, **tags):
        mlflow.log_params({
            "model_type": model_type,
            "scaler": scaler,
            "target_transform": target_transform,
            "weight_half_life": str(weight_half_life),
            **{f"hp_{k}": v for k, v in model_params.items()},
        })
        mlflow.log_metrics(cv_metrics)


# ── Tree tuning ──────────────────────────────────────────────────


def tune_tree_model(
    dataset_path: Path,
    model_type: str,
    *,
    feature_version: str,
    cv_folds: int = SEARCH_CV_FOLDS,
    cv_mode: str = "expanding",
    holdout_days: int = HOLDOUT_DAYS,
) -> dict[str, Any]:
    """Two-stage grid search for a tree model. Returns the best config.

    Stage 1: pin weight by sweeping ``WEIGHT_HALF_LIVES`` against the
    probe config.

    Stage 2: walk ``PRICE_TREE_GRID[model_type]`` with the winning weight.

    The merged probe + full-grid results are returned, plus the winning
    config so the caller (§5c.4) can retrain it for the ensemble.
    """
    if model_type not in PRICE_TREE_WEIGHT_PROBE:
        raise ValueError(f"Tree tuning unsupported for {model_type!r}")

    X, y = load_dataset(dataset_path)
    pool_idx, _holdout_idx = carve_holdout(X.index, holdout_days)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    cv = TimeSeriesSplitter(n_splits=cv_folds, mode=cv_mode)

    probe = PRICE_TREE_WEIGHT_PROBE[model_type]

    # Stage 1 — weight pinning.
    weight_study_name = f"{feature_version}__{model_type}__stage1_weight"
    weight_study = _grid_study(
        weight_study_name,
        {"weight_half_life_idx": list(range(len(WEIGHT_HALF_LIVES)))},
    )

    def stage1_obj(trial: optuna.Trial) -> float:
        wh_idx = trial.suggest_int(
            "weight_half_life_idx", 0, len(WEIGHT_HALF_LIVES) - 1,
        )
        weight = WEIGHT_HALF_LIVES[wh_idx]
        metrics = _evaluate_config(
            X_pool, y_pool, cv,
            model_type=model_type,
            model_params=probe,
            scaler="none",
            target_transform="none",
            weight_half_life=weight,
        )
        _log_trial_to_mlflow(
            experiment="price_model_training",
            feature_version=feature_version,
            dataset_path=dataset_path,
            model_type=model_type,
            tuning_step="weight_pin",
            model_params=probe,
            scaler="none",
            target_transform="none",
            weight_half_life=weight,
            cv_metrics=metrics,
        )
        return metrics["cv_mae"]

    weight_study.optimize(stage1_obj, n_trials=len(WEIGHT_HALF_LIVES))
    best_weight_idx = weight_study.best_params["weight_half_life_idx"]
    best_weight = WEIGHT_HALF_LIVES[best_weight_idx]
    logger.info(
        f"[{model_type}] stage1 winner: weight_half_life={best_weight} "
        f"(cv_mae={weight_study.best_value:.3f})"
    )

    # Stage 2 — hyperparam grid with weight fixed.
    grid_configs = PRICE_TREE_GRID[model_type]
    grid_study_name = f"{feature_version}__{model_type}__stage2_grid"
    grid_study = _grid_study(
        grid_study_name,
        {"grid_idx": list(range(len(grid_configs)))},
    )

    def stage2_obj(trial: optuna.Trial) -> float:
        idx = trial.suggest_int("grid_idx", 0, len(grid_configs) - 1)
        # Merge probe defaults with the grid override.
        params = {**probe, **grid_configs[idx]}
        metrics = _evaluate_config(
            X_pool, y_pool, cv,
            model_type=model_type,
            model_params=params,
            scaler="none",
            target_transform="none",
            weight_half_life=best_weight,
        )
        _log_trial_to_mlflow(
            experiment="price_model_training",
            feature_version=feature_version,
            dataset_path=dataset_path,
            model_type=model_type,
            tuning_step="hyperparam_grid",
            model_params=params,
            scaler="none",
            target_transform="none",
            weight_half_life=best_weight,
            cv_metrics=metrics,
        )
        return metrics["cv_mae"]

    grid_study.optimize(stage2_obj, n_trials=len(grid_configs))
    best_idx = grid_study.best_params["grid_idx"]
    best_params = {**probe, **grid_configs[best_idx]}

    return {
        "model_type": model_type,
        "model_params": best_params,
        "scaler": "none",
        "target_transform": "none",
        "weight_half_life": best_weight,
        "cv_mae": grid_study.best_value,
        "stage1_study": weight_study_name,
        "stage2_study": grid_study_name,
    }


# ── Linear tuning ────────────────────────────────────────────────


def _linear_search_space(model_type: str) -> dict[str, list[Any]]:
    """Build the exhaustive grid for a linear model."""
    alphas = LINEAR_ALPHA_GRID[model_type]
    return {
        "preproc_idx": list(range(len(LINEAR_PREPROCESSING_GRID))),
        "alpha_idx": list(range(len(alphas))),
    }


def tune_linear_model(
    dataset_path: Path,
    model_type: str,
    *,
    feature_version: str,
    cv_folds: int = SEARCH_CV_FOLDS,
    cv_mode: str = "expanding",
    holdout_days: int = HOLDOUT_DAYS,
) -> dict[str, Any]:
    """Exhaustive grid over preprocessing × alpha."""
    if model_type not in LINEAR_ALPHA_GRID:
        raise ValueError(f"Linear tuning unsupported for {model_type!r}")

    X, y = load_dataset(dataset_path)
    pool_idx, _holdout_idx = carve_holdout(X.index, holdout_days)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    cv = TimeSeriesSplitter(n_splits=cv_folds, mode=cv_mode)

    search_space = _linear_search_space(model_type)
    study_name = f"{feature_version}__{model_type}__linear_grid"
    study = _grid_study(study_name, search_space)
    alphas = LINEAR_ALPHA_GRID[model_type]

    def linear_obj(trial: optuna.Trial) -> float:
        preproc_idx = trial.suggest_int("preproc_idx", 0, len(LINEAR_PREPROCESSING_GRID) - 1)
        alpha_idx = trial.suggest_int("alpha_idx", 0, len(alphas) - 1)
        scaler, weight_half_life = LINEAR_PREPROCESSING_GRID[preproc_idx]
        target_transform = "none"
        params: dict[str, Any] = {"alpha": alphas[alpha_idx]}

        metrics = _evaluate_config(
            X_pool, y_pool, cv,
            model_type=model_type,
            model_params=params,
            scaler=scaler,
            target_transform=target_transform,
            weight_half_life=weight_half_life,
        )
        _log_trial_to_mlflow(
            experiment="price_model_training",
            feature_version=feature_version,
            dataset_path=dataset_path,
            model_type=model_type,
            tuning_step="linear_grid",
            model_params=params,
            scaler=scaler,
            target_transform=target_transform,
            weight_half_life=weight_half_life,
            cv_metrics=metrics,
        )
        return metrics["cv_mae"]

    # n_trials = full grid size.
    n_trials = int(np.prod([len(v) for v in search_space.values()]))
    study.optimize(linear_obj, n_trials=n_trials)

    # Reconstruct the winning config.
    best = study.best_params
    scaler, weight_half_life = LINEAR_PREPROCESSING_GRID[best["preproc_idx"]]
    target_transform = "none"
    best_params: dict[str, Any] = {"alpha": alphas[best["alpha_idx"]]}

    return {
        "model_type": model_type,
        "model_params": best_params,
        "scaler": scaler,
        "target_transform": target_transform,
        "weight_half_life": weight_half_life,
        "cv_mae": study.best_value,
        "study": study_name,
    }


# ── Orchestrator ─────────────────────────────────────────────────


PRICE_TREE_TYPES = ("LGBMRegressor", "XGBRegressor", "CatBoostRegressor")
PRICE_LINEAR_TYPES = ("Ridge", "Lasso")


def tune_all_price_models(
    dataset_path: Path,
    feature_version: str,
    *,
    tree_types: tuple[str, ...] = PRICE_TREE_TYPES,
    linear_types: tuple[str, ...] = PRICE_LINEAR_TYPES,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Tune every supported model on one dataset. Returns the winners."""
    winners: dict[str, dict[str, Any]] = {}
    for mt in tree_types:
        winners[mt] = tune_tree_model(
            dataset_path, mt, feature_version=feature_version, **kwargs,
        )
    for mt in linear_types:
        winners[mt] = tune_linear_model(
            dataset_path, mt, feature_version=feature_version, **kwargs,
        )
    return winners
