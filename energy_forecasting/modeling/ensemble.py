"""Price-ensemble construction and diagnostic comparison helpers.

Production intentionally follows EP's simple blend contract: select the category
floor, fit inverse-MAE weights on the recent holdout, and deploy those weights.
The richer method registry, stackers, floor variants, and single-model baselines
are retained for research diagnostics only; production must not select from that
bakeoff table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge

from energy_forecasting.config.modeling import (
    ENSEMBLE_CATEGORY_MATCHERS,
    ENSEMBLE_MAX_ALIGNMENT_DROP_FRACTION,
    ENSEMBLE_METHODS,
    ENSEMBLE_MIN_MEMBER_WEIGHT,
)
from energy_forecasting.modeling.metrics import calculate_metrics
from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking


@dataclass
class WeightEnsemble:
    """Linear combination of base-model predictions."""

    method: str
    weights: np.ndarray
    model_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, preds_matrix: np.ndarray) -> np.ndarray:
        preds_matrix = _as_2d_array(preds_matrix, self.model_names)
        return preds_matrix @ self.weights


@dataclass
class StackEnsemble:
    """Meta-learner trained on OOF base-model predictions."""

    method: str
    meta_learner: Any
    model_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, preds_matrix: np.ndarray) -> np.ndarray:
        preds_matrix = _as_2d_array(preds_matrix, self.model_names)
        return self.meta_learner.predict(preds_matrix)


Ensemble = WeightEnsemble | StackEnsemble
PredictionLoader = Callable[[str], tuple[pd.DataFrame, pd.DataFrame]]


@dataclass
class ProductionEnsembleResult:
    method: str
    ensemble: WeightEnsemble
    metrics: dict[str, Any]
    comparison: pd.DataFrame
    selected_model_runs: list[Any]
    selected_models: list[str]
    candidate_metrics: pd.DataFrame
    alignment_metadata: dict[str, Any]


# -- Shared helpers ---------------------------------------------------------


def _as_2d_array(preds: Any, expected_models: list[str]) -> np.ndarray:
    """Coerce predictions into ``(n_samples, n_models)`` with the right ordering."""
    if isinstance(preds, pd.DataFrame):
        missing = [m for m in expected_models if m not in preds.columns]
        if missing:
            raise KeyError(f"preds_matrix missing model columns: {missing}")
        return preds[expected_models].to_numpy(dtype=float)
    arr = np.asarray(preds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(expected_models):
        raise ValueError(
            f"preds_matrix must be (n_samples, {len(expected_models)}); got {arr.shape}"
        )
    return arr


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Project to the probability simplex."""
    weights = np.clip(weights.astype(float), 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _apply_min_weight_floor(weights: np.ndarray, floor: float) -> np.ndarray:
    """Apply a true minimum per-member floor, then allocate the remainder."""
    weights = _normalise(weights)
    n = len(weights)
    if floor <= 0:
        return weights
    if floor * n >= 1.0:
        return np.full(n, 1.0 / n)
    residual = 1.0 - floor * n
    return np.full(n, floor) + residual * weights


def _with_weight_floor(method: str, base_factory: Callable[[pd.DataFrame, pd.Series], WeightEnsemble]):
    def fit(preds_oof: pd.DataFrame, y_oof: pd.Series) -> WeightEnsemble:
        base = base_factory(preds_oof, y_oof)
        pre_floor = _normalise(base.weights)
        weights = _apply_min_weight_floor(pre_floor, ENSEMBLE_MIN_MEMBER_WEIGHT)
        metadata = dict(base.metadata)
        metadata.update(
            {
                "base_method": base.method,
                "pre_floor_weights": pre_floor.tolist(),
                "post_floor_weights": weights.tolist(),
                "min_member_weight": ENSEMBLE_MIN_MEMBER_WEIGHT,
            }
        )
        return WeightEnsemble(
            method=method,
            weights=weights,
            model_names=list(base.model_names),
            metadata=metadata,
        )

    return fit


# -- Weight-based methods ---------------------------------------------------


def fit_simple_average(preds_oof: pd.DataFrame, y_oof: pd.Series) -> WeightEnsemble:
    n = preds_oof.shape[1]
    return WeightEnsemble(
        method="simple_average",
        weights=np.full(n, 1.0 / n),
        model_names=list(preds_oof.columns),
    )


def fit_inverse_mae(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    eps: float = 1e-6,
) -> WeightEnsemble:
    """Fit ``w_i proportional to 1 / (OOF_MAE_i + eps)``."""
    y_true = y_oof.to_numpy(dtype=float)
    maes = np.array([_mae(y_true, preds_oof[m].to_numpy(float)) for m in preds_oof.columns])
    raw = 1.0 / (maes + eps)
    return WeightEnsemble(
        method="inverse_mae",
        weights=_normalise(raw),
        model_names=list(preds_oof.columns),
        metadata={"per_model_mae": maes.tolist()},
    )


def fit_inverse_rmse(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    eps: float = 1e-6,
) -> WeightEnsemble:
    """Fit ``w_i proportional to 1 / (OOF_RMSE_i + eps)``."""
    y_true = y_oof.to_numpy(dtype=float)
    rmses = np.array([_rmse(y_true, preds_oof[m].to_numpy(float)) for m in preds_oof.columns])
    raw = 1.0 / (rmses + eps)
    return WeightEnsemble(
        method="inverse_rmse",
        weights=_normalise(raw),
        model_names=list(preds_oof.columns),
        metadata={"per_model_rmse": rmses.tolist()},
    )


def fit_top_k_trimmed(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    keep: int | None = None,
    eps: float = 1e-6,
) -> WeightEnsemble:
    """Drop worst OOF-MAE models, then inverse-MAE weight the remainder."""
    y_true = y_oof.to_numpy(dtype=float)
    maes = np.array([_mae(y_true, preds_oof[m].to_numpy(float)) for m in preds_oof.columns])
    n_models = len(maes)
    if keep is None:
        keep = min(6, n_models)
    keep = min(keep, n_models)

    weights = np.zeros(n_models)
    top_idx = np.argsort(maes)[:keep]
    inv = 1.0 / (maes[top_idx] + eps)
    weights[top_idx] = inv / inv.sum()
    return WeightEnsemble(
        method="top_k_trimmed",
        weights=weights,
        model_names=list(preds_oof.columns),
        metadata={"keep": int(keep), "per_model_mae": maes.tolist()},
    )


def fit_slsqp(preds_oof: pd.DataFrame, y_oof: pd.Series) -> WeightEnsemble:
    """Constrained OOF MAE minimisation: sum(weights)=1, weights>=0."""
    from scipy.optimize import minimize

    P = preds_oof.to_numpy(dtype=float)
    y = y_oof.to_numpy(dtype=float)
    n = P.shape[1]

    def loss(w: np.ndarray) -> float:
        return _mae(y, P @ w)

    init = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(loss, init, method="SLSQP", bounds=bounds, constraints=constraints)
    return WeightEnsemble(
        method="slsqp_optimized",
        weights=_normalise(res.x),
        model_names=list(preds_oof.columns),
        metadata={"converged": bool(res.success), "fit_oof_mae": float(res.fun)},
    )


def fit_greedy_forward(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    max_models: int | None = None,
    diversity_weight: float = 0.0,
) -> WeightEnsemble:
    """Greedy ensemble selection on OOF predictions."""
    P = preds_oof.to_numpy(dtype=float)
    y = y_oof.to_numpy(dtype=float)
    n_samples, n_models = P.shape
    if max_models is None:
        max_models = 50 * n_models

    counts = np.zeros(n_models, dtype=int)
    running_sum = np.zeros(n_samples)
    best_mae = float("inf")
    for _ in range(max_models):
        cand_maes = np.empty(n_models)
        for j in range(n_models):
            cand_sum = running_sum + P[:, j]
            ensemble_pred = cand_sum / (counts.sum() + 1)
            mae = _mae(y, ensemble_pred)
            if diversity_weight > 0 and counts.sum() > 0:
                current = running_sum / counts.sum()
                corr = np.corrcoef(current, P[:, j])[0, 1]
                mae -= diversity_weight * (1.0 - abs(corr))
            cand_maes[j] = mae
        best_j = int(np.argmin(cand_maes))
        counts[best_j] += 1
        running_sum += P[:, best_j]
        if cand_maes[best_j] >= best_mae - 1e-6:
            break
        best_mae = float(cand_maes[best_j])

    return WeightEnsemble(
        method="diversity_regularized" if diversity_weight > 0 else "greedy_forward",
        weights=_normalise(counts.astype(float)),
        model_names=list(preds_oof.columns),
        metadata={"counts": counts.tolist(), "fit_oof_mae": best_mae},
    )


def fit_hill_climbing(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    max_iter: int = 200,
) -> WeightEnsemble:
    """Start from greedy, then accept count add/drop moves that improve OOF MAE."""
    greedy = fit_greedy_forward(preds_oof, y_oof)
    P = preds_oof.to_numpy(dtype=float)
    y = y_oof.to_numpy(dtype=float)
    n_models = P.shape[1]
    counts = np.asarray(greedy.metadata["counts"], dtype=int)

    def score(c: np.ndarray) -> float:
        total = c.sum()
        if total == 0:
            return float("inf")
        return _mae(y, P @ (c / total))

    best = score(counts)
    for _ in range(max_iter):
        improved = False
        for j in range(n_models):
            for delta in (-1, 1):
                if counts[j] + delta < 0:
                    continue
                cand = counts.copy()
                cand[j] += delta
                cand_score = score(cand)
                if cand_score < best - 1e-9:
                    counts = cand
                    best = cand_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return WeightEnsemble(
        method="hill_climbing",
        weights=_normalise(counts.astype(float)),
        model_names=list(preds_oof.columns),
        metadata={"counts": counts.tolist(), "fit_oof_mae": best},
    )


def fit_simulated_annealing(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    n_iter: int = 1500,
    initial_temp: float = 1.0,
    cooling: float = 0.995,
    rng_seed: int = 0,
) -> WeightEnsemble:
    """Random-restart hill climber with a temperature schedule."""
    P = preds_oof.to_numpy(dtype=float)
    y = y_oof.to_numpy(dtype=float)
    n_models = P.shape[1]
    rng = np.random.default_rng(rng_seed)

    counts = np.ones(n_models, dtype=int)

    def score(c: np.ndarray) -> float:
        total = c.sum()
        if total == 0:
            return float("inf")
        return _mae(y, P @ (c / total))

    current = score(counts)
    best, best_counts = current, counts.copy()
    temp = initial_temp
    for _ in range(n_iter):
        j = rng.integers(0, n_models)
        delta = rng.choice([-1, 1])
        if counts[j] + delta < 0:
            continue
        counts[j] += delta
        new_score = score(counts)
        delta_score = new_score - current
        if delta_score < 0 or rng.random() < np.exp(-delta_score / max(temp, 1e-9)):
            current = new_score
            if new_score < best:
                best = new_score
                best_counts = counts.copy()
        else:
            counts[j] -= delta
        temp *= cooling

    return WeightEnsemble(
        method="simulated_annealing",
        weights=_normalise(best_counts.astype(float)),
        model_names=list(preds_oof.columns),
        metadata={"counts": best_counts.tolist(), "fit_oof_mae": best},
    )


def fit_diversity_regularized(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    diversity_weight: float = 0.05,
) -> WeightEnsemble:
    return fit_greedy_forward(preds_oof, y_oof, diversity_weight=diversity_weight)


def fit_slsqp_floor_2pct(preds_oof: pd.DataFrame, y_oof: pd.Series) -> WeightEnsemble:
    return _with_weight_floor("slsqp_floor_2pct", fit_slsqp)(preds_oof, y_oof)


def fit_greedy_forward_floor_2pct(preds_oof: pd.DataFrame, y_oof: pd.Series) -> WeightEnsemble:
    return _with_weight_floor("greedy_forward_floor_2pct", fit_greedy_forward)(preds_oof, y_oof)


def fit_hill_climbing_floor_2pct(preds_oof: pd.DataFrame, y_oof: pd.Series) -> WeightEnsemble:
    return _with_weight_floor("hill_climbing_floor_2pct", fit_hill_climbing)(preds_oof, y_oof)


def fit_simulated_annealing_floor_2pct(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
) -> WeightEnsemble:
    return _with_weight_floor("simulated_annealing_floor_2pct", fit_simulated_annealing)(
        preds_oof,
        y_oof,
    )


# -- Stacking methods -------------------------------------------------------


def fit_stacking_ridge(preds_oof: pd.DataFrame, y_oof: pd.Series) -> StackEnsemble:
    meta = Ridge(positive=True, alpha=1.0)
    meta.fit(preds_oof.to_numpy(float), y_oof.to_numpy(float))
    return StackEnsemble(
        method="stacking_ridge",
        meta_learner=meta,
        model_names=list(preds_oof.columns),
    )


def fit_stacking_lgbm(preds_oof: pd.DataFrame, y_oof: pd.Series) -> StackEnsemble:
    from lightgbm import LGBMRegressor

    meta = LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="mae",
        verbose=-1,
        n_jobs=-1,
    )
    meta.fit(preds_oof.to_numpy(float), y_oof.to_numpy(float))
    return StackEnsemble(
        method="stacking_lgbm",
        meta_learner=meta,
        model_names=list(preds_oof.columns),
    )


METHOD_FACTORIES: dict[str, Callable[[pd.DataFrame, pd.Series], Ensemble]] = {
    "simple_average": fit_simple_average,
    "inverse_mae": fit_inverse_mae,
    "inverse_rmse": fit_inverse_rmse,
    "top_k_trimmed": fit_top_k_trimmed,
    "slsqp_optimized": fit_slsqp,
    "slsqp_floor_2pct": fit_slsqp_floor_2pct,
    "greedy_forward": fit_greedy_forward,
    "greedy_forward_floor_2pct": fit_greedy_forward_floor_2pct,
    "hill_climbing": fit_hill_climbing,
    "hill_climbing_floor_2pct": fit_hill_climbing_floor_2pct,
    "simulated_annealing": fit_simulated_annealing,
    "simulated_annealing_floor_2pct": fit_simulated_annealing_floor_2pct,
    "diversity_regularized": fit_diversity_regularized,
    "stacking_ridge": fit_stacking_ridge,
    "stacking_lgbm": fit_stacking_lgbm,
}
PRODUCTION_WEIGHT_METHODS = frozenset(
    {
        "simple_average",
        "inverse_mae",
        "inverse_rmse",
        "top_k_trimmed",
        "slsqp_optimized",
        "slsqp_floor_2pct",
        "greedy_forward",
        "greedy_forward_floor_2pct",
        "hill_climbing",
        "hill_climbing_floor_2pct",
        "simulated_annealing",
        "simulated_annealing_floor_2pct",
        "diversity_regularized",
    }
)
DIAGNOSTIC_STACKING_METHODS = frozenset({"stacking_ridge", "stacking_lgbm"})
OOF_FIT_METHODS = PRODUCTION_WEIGHT_METHODS | DIAGNOSTIC_STACKING_METHODS
HOLDOUT_FIT_METHODS = frozenset()


def fit_ensemble(method: str, preds: pd.DataFrame, y: pd.Series) -> Ensemble:
    if method not in METHOD_FACTORIES:
        raise ValueError(
            f"Unknown ensemble method {method!r}. Available: {sorted(METHOD_FACTORIES)}"
        )
    return METHOD_FACTORIES[method](preds, y)


# -- Prediction loading and alignment --------------------------------------


def fetch_prediction_artifacts(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OOF + holdout prediction parquets for one MLflow run."""
    ensure_mlflow_tracking()
    client = mlflow.MlflowClient()
    artifact_dir = Path(client.download_artifacts(run_id, "predictions"))
    oof = pd.read_parquet(artifact_dir / "oof_predictions.parquet")
    holdout = pd.read_parquet(artifact_dir / "holdout_predictions.parquet")
    return oof, holdout


def _prediction_name(model_run: Any) -> str:
    return str(getattr(model_run, "name"))


def _prediction_run_id(model_run: Any) -> str:
    return str(getattr(model_run, "run_id"))


def _check_prediction_frame(name: str, frame: pd.DataFrame, window: str) -> None:
    missing = {"y_true", "y_pred"} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} {window} predictions missing columns: {sorted(missing)}")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} {window} predictions have duplicate indexes")
    if frame.empty:
        raise ValueError(f"{name} {window} predictions are empty")


def _common_index(
    frames: dict[str, pd.DataFrame],
    *,
    window: str,
    max_drop_fraction: float,
) -> tuple[pd.Index, dict[str, Any]]:
    names = list(frames)
    common = frames[names[0]].index
    for name in names[1:]:
        common = common.intersection(frames[name].index)
    common = common.sort_values()
    if common.empty:
        raise ValueError(f"Common {window} prediction index is empty")

    input_counts = {name: int(len(frame)) for name, frame in frames.items()}
    min_input_count = min(input_counts.values())
    dropped_from_min = min_input_count - len(common)
    drop_fraction = dropped_from_min / min_input_count if min_input_count else 1.0
    if drop_fraction > max_drop_fraction:
        raise ValueError(
            f"Common {window} prediction index drops {drop_fraction:.1%} of the smallest "
            f"input ({len(common)} common rows vs {min_input_count} min input rows)"
        )

    return common, {
        f"{window}_input_row_counts": input_counts,
        f"{window}_row_count": int(len(common)),
        f"{window}_min_input_row_count": int(min_input_count),
        f"{window}_intersection_drop_fraction": float(drop_fraction),
    }


def _assert_y_true_agrees(
    frames: dict[str, pd.DataFrame],
    common: pd.Index,
    *,
    window: str,
) -> pd.Series:
    names = list(frames)
    ref = frames[names[0]].loc[common, "y_true"]
    ref_values = ref.to_numpy(dtype=float)
    for name in names[1:]:
        values = frames[name].loc[common, "y_true"].to_numpy(dtype=float)
        if not np.allclose(ref_values, values, equal_nan=True):
            raise ValueError(f"y_true mismatch across models on common {window} index: {name}")
    return ref


def validate_prediction_alignment(
    oof_frames: dict[str, pd.DataFrame],
    holdout_frames: dict[str, pd.DataFrame],
    *,
    max_drop_fraction: float = ENSEMBLE_MAX_ALIGNMENT_DROP_FRACTION,
) -> tuple[pd.Index, pd.Index, pd.Series, pd.Series, dict[str, Any]]:
    """Validate OOF/holdout prediction frames before fitting or scoring."""
    if not oof_frames:
        raise ValueError("No OOF prediction frames supplied")
    if set(oof_frames) != set(holdout_frames):
        raise ValueError("OOF and holdout prediction frames must have the same model names")

    for name in oof_frames:
        _check_prediction_frame(name, oof_frames[name], "OOF")
        _check_prediction_frame(name, holdout_frames[name], "holdout")

    common_oof, oof_meta = _common_index(
        oof_frames,
        window="oof",
        max_drop_fraction=max_drop_fraction,
    )
    common_holdout, holdout_meta = _common_index(
        holdout_frames,
        window="holdout",
        max_drop_fraction=max_drop_fraction,
    )
    y_oof = _assert_y_true_agrees(oof_frames, common_oof, window="OOF")
    y_holdout = _assert_y_true_agrees(holdout_frames, common_holdout, window="holdout")
    metadata = {**oof_meta, **holdout_meta}
    logger.info(
        "Aligned price ensemble predictions: "
        f"OOF rows={metadata['oof_row_count']}, holdout rows={metadata['holdout_row_count']}"
    )
    return common_oof, common_holdout, y_oof, y_holdout, metadata


def stack_model_predictions(
    model_runs: list[Any],
    *,
    prediction_loader: PredictionLoader | None = None,
    max_drop_fraction: float = ENSEMBLE_MAX_ALIGNMENT_DROP_FRACTION,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load, validate, and align base-model OOF and holdout predictions."""
    prediction_loader = prediction_loader or fetch_prediction_artifacts
    oof_frames: dict[str, pd.DataFrame] = {}
    holdout_frames: dict[str, pd.DataFrame] = {}
    for model_run in model_runs:
        name = _prediction_name(model_run)
        oof, holdout = prediction_loader(_prediction_run_id(model_run))
        oof_frames[name] = oof
        holdout_frames[name] = holdout

    common_oof, common_holdout, y_oof, y_holdout, metadata = validate_prediction_alignment(
        oof_frames,
        holdout_frames,
        max_drop_fraction=max_drop_fraction,
    )
    names = [_prediction_name(mr) for mr in model_runs]
    preds_oof = pd.DataFrame(
        {name: oof_frames[name].loc[common_oof, "y_pred"] for name in names},
        index=common_oof,
    )
    preds_holdout = pd.DataFrame(
        {name: holdout_frames[name].loc[common_holdout, "y_pred"] for name in names},
        index=common_holdout,
    )
    return preds_oof, y_oof, preds_holdout, y_holdout, metadata


# -- Candidate selection ----------------------------------------------------


def _category_for_model(model_type: str) -> str | None:
    for category, matchers in ENSEMBLE_CATEGORY_MATCHERS.items():
        if any(matcher == model_type or matcher in model_type for matcher in matchers):
            return category
    return None


def select_final_models(
    model_runs: list[Any],
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    strict_categories: bool = False,
) -> tuple[list[Any], pd.DataFrame]:
    """Keep best OOF-MAE and best OOF-RMSE model per available category."""
    run_by_name = {_prediction_name(mr): mr for mr in model_runs}
    rows: list[dict[str, Any]] = []
    y = y_oof.to_numpy(dtype=float)
    for name in preds_oof.columns:
        model_run = run_by_name[name]
        model_type = str(getattr(model_run, "model_type", ""))
        category = _category_for_model(model_type)
        pred = preds_oof[name].to_numpy(dtype=float)
        rows.append(
            {
                "model": name,
                "model_type": model_type,
                "feature_version": getattr(model_run, "feature_version", None),
                "category": category,
                "oof_mae": _mae(y, pred),
                "oof_rmse": _rmse(y, pred),
                "selected_by": "",
            }
        )

    metrics = pd.DataFrame(rows)
    missing_categories = [
        category
        for category in ENSEMBLE_CATEGORY_MATCHERS
        if category not in set(metrics["category"].dropna())
    ]
    if missing_categories:
        message = f"Missing ensemble model categories: {missing_categories}"
        if strict_categories:
            raise ValueError(message)
        logger.warning(message)

    selected: set[str] = set()
    selected_by: dict[str, set[str]] = {}
    for category, group in metrics.dropna(subset=["category"]).groupby("category"):
        mae_name = str(group.sort_values("oof_mae").iloc[0]["model"])
        rmse_order = group.sort_values("oof_rmse")
        rmse_name = str(rmse_order.iloc[0]["model"])
        if rmse_name == mae_name and len(rmse_order) > 1:
            rmse_name = str(rmse_order.iloc[1]["model"])
        selected.add(mae_name)
        selected.add(rmse_name)
        selected_by.setdefault(mae_name, set()).add(f"{category}:oof_mae")
        selected_by.setdefault(rmse_name, set()).add(f"{category}:oof_rmse")

    if not selected:
        raise ValueError("No ensemble candidates selected; check model categories")

    metrics["selected"] = metrics["model"].isin(selected)
    metrics["selected_by"] = metrics["model"].map(
        lambda name: ",".join(sorted(selected_by.get(str(name), set())))
    )
    selected_runs = [mr for mr in model_runs if _prediction_name(mr) in selected]
    logger.info(
        "Selected final ensemble candidates: "
        + ", ".join(_prediction_name(mr) for mr in selected_runs)
    )
    return selected_runs, metrics.sort_values(["category", "oof_mae"]).reset_index(drop=True)


# -- Compare, select, and build --------------------------------------------


def _method_kind(method: str) -> str:
    if method in PRODUCTION_WEIGHT_METHODS:
        return "weight"
    if method in DIAGNOSTIC_STACKING_METHODS:
        return "stacker"
    if method.startswith("single::"):
        return "single"
    return "unknown"


def compare_ensemble_methods(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    preds_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    methods: Iterable[str] | None = None,
    include_base_models: bool = True,
) -> pd.DataFrame:
    """Fit methods on OOF predictions and evaluate them on holdout predictions."""
    methods = list(methods) if methods is not None else list(ENSEMBLE_METHODS)
    if list(preds_oof.columns) != list(preds_holdout.columns):
        raise ValueError("preds_oof and preds_holdout must share columns in the same order.")

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Ensemble] = {}
    for method in methods:
        try:
            ens = fit_ensemble(method, preds_oof, y_oof)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Ensemble {method!r} failed during fit: {exc}")
            continue
        fitted[method] = ens
        preds = ens.predict(preds_holdout)
        metrics = calculate_metrics(y_holdout, preds)
        kind = _method_kind(method)
        rows.append(
            {
                "method": method,
                "method_kind": kind,
                "fit_window": "oof",
                "metric_window": "holdout",
                "eligible_for_production": isinstance(ens, WeightEnsemble)
                and kind in {"weight", "single"},
                **metrics,
            }
        )

    if include_base_models:
        for model_name in preds_oof.columns:
            method_label = f"single::{model_name}"
            weights = np.array(
                [1.0 if c == model_name else 0.0 for c in preds_oof.columns],
                dtype=float,
            )
            ens = WeightEnsemble(
                method=method_label,
                weights=weights,
                model_names=list(preds_oof.columns),
                metadata={"single_model": model_name},
            )
            fitted[method_label] = ens
            preds = ens.predict(preds_holdout)
            metrics = calculate_metrics(y_holdout, preds)
            rows.append(
                {
                    "method": method_label,
                    "method_kind": "single",
                    "fit_window": "none",
                    "metric_window": "holdout",
                    "eligible_for_production": True,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    df.attrs["fitted"] = fitted
    return df


def select_best_ensemble(comparison: pd.DataFrame) -> tuple[str, WeightEnsemble, dict[str, Any]]:
    """Pick the lowest-MAE row that is eligible for production."""
    if comparison.empty:
        raise ValueError("No ensemble candidates to choose from.")
    fitted: dict[str, Ensemble] = comparison.attrs.get("fitted", {})
    if not fitted:
        raise ValueError(
            "comparison.attrs['fitted'] missing. Call compare_ensemble_methods first."
        )
    eligible = comparison[comparison.get("eligible_for_production", False) == True]  # noqa: E712
    if eligible.empty:
        raise ValueError("No production-eligible ensemble candidates to choose from.")

    winner_row = eligible.sort_values("mae").iloc[0]
    method = str(winner_row["method"])
    ensemble = fitted[method]
    if not isinstance(ensemble, WeightEnsemble):
        raise TypeError(f"Selected production ensemble {method!r} is not weight-based")
    metrics = {k: v for k, v in winner_row.items() if k != "method"}
    return method, ensemble, metrics


def _calibrate_holdout_intervals(
    ensemble: WeightEnsemble,
    preds_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> dict[str, float]:
    from energy_forecasting.modeling.intervals import (
        calibrate_ensemble_intervals,
        predict_ensemble_intervals,
    )

    holdout_pred = ensemble.predict(preds_holdout)
    conformal_quantile = calibrate_ensemble_intervals(y_holdout, holdout_pred)
    pi_lower, pi_upper = predict_ensemble_intervals(holdout_pred, conformal_quantile)
    coverage = float(
        np.mean((y_holdout.to_numpy() >= pi_lower) & (y_holdout.to_numpy() <= pi_upper))
    )
    pi_width = float(np.mean(pi_upper - pi_lower))
    return {
        "conformal_quantile": float(conformal_quantile),
        "pi_coverage": coverage,
        "pi_width": pi_width,
    }


def build_production_ensemble(
    model_runs: list[Any],
    *,
    prediction_loader: PredictionLoader | None = None,
    strict_categories: bool = False,
    candidate_model_names: set[str] | None = None,
    methods: Iterable[str] | None = None,
) -> ProductionEnsembleResult:
    """Build the EP-faithful deployable price ensemble from base-model runs.

    ``methods`` is accepted for backwards-compatible callers but intentionally
    ignored: production always uses inverse-MAE weights fitted on the recent
    holdout. Use ``compare_ensemble_methods`` from the research script for the
    diagnostic bakeoff.
    """
    if methods is not None:
        logger.warning("Ignoring production ensemble methods override; using inverse_mae")

    preds_oof, y_oof, preds_holdout, y_holdout, alignment = stack_model_predictions(
        model_runs,
        prediction_loader=prediction_loader,
    )

    if candidate_model_names is None:
        selected_runs, candidate_metrics = select_final_models(
            model_runs,
            preds_oof,
            y_oof,
            strict_categories=strict_categories,
        )
        selected_names = [_prediction_name(mr) for mr in selected_runs]
        candidate_selection_window = "oof"
    else:
        selected_names = [name for name in preds_holdout.columns if name in candidate_model_names]
        if not selected_names:
            raise ValueError("No requested production members found in prediction columns")
        selected_runs = [mr for mr in model_runs if _prediction_name(mr) in set(selected_names)]
        candidate_metrics = pd.DataFrame(
            {
                "model": selected_names,
                "selected": True,
                "selected_by": "fixed_production_member",
            }
        )
        candidate_selection_window = "fixed_production_member"

    preds_holdout_final = preds_holdout[selected_names]
    ensemble = fit_inverse_mae(preds_holdout_final, y_holdout)
    y_pred = ensemble.predict(preds_holdout_final)
    metrics = calculate_metrics(y_holdout, y_pred)
    metrics.update(_calibrate_holdout_intervals(ensemble, preds_holdout_final, y_holdout))
    metrics.update(
        {
            "method_kind": "weight",
            "fit_window": "recent_holdout",
            "metric_window": "recent_holdout_in_sample",
            "selection_fit_window": "recent_holdout",
            "candidate_selection_window": candidate_selection_window,
            "conformal_calibration_window": "recent_holdout_in_sample",
            "selection_metric": "fixed_inverse_mae",
            "oof_row_count": int(alignment["oof_row_count"]),
            "holdout_row_count": int(alignment["holdout_row_count"]),
        }
    )

    comparison = pd.DataFrame()
    comparison.attrs["alignment_metadata"] = alignment
    comparison.attrs["candidate_models"] = selected_names

    return ProductionEnsembleResult(
        method="inverse_mae",
        ensemble=ensemble,
        metrics=_json_native(metrics),
        comparison=comparison,
        selected_model_runs=selected_runs,
        selected_models=selected_names,
        candidate_metrics=candidate_metrics,
        alignment_metadata=alignment,
    )


def _json_native(value: Any) -> Any:
    """Recursively coerce numpy/pandas scalars into JSON-native values."""
    if isinstance(value, dict):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_native(v) for v in value]
    if isinstance(value, tuple):
        return [_json_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ensemble_config_dict(
    ensemble: Ensemble,
    *,
    base_runs: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serialisable summary written to ensemble_config.json."""
    metrics = _json_native(metrics)
    if isinstance(ensemble, WeightEnsemble):
        ensemble_section: dict[str, Any] = {
            "method": ensemble.method,
            "weights_fit_window": "recent_holdout",
            "weights": {
                name: float(w)
                for name, w in zip(ensemble.model_names, ensemble.weights, strict=True)
            },
        }
    else:
        ensemble_section = {
            "method": ensemble.method,
            "weights_fit_window": "diagnostic_oof",
            "meta_learner": type(ensemble.meta_learner).__name__,
            "model_names": list(ensemble.model_names),
        }

    return {
        "ensemble": ensemble_section,
        "metrics": metrics,
        "conformal_quantile": metrics.get("conformal_quantile"),
        "pi_coverage": metrics.get("pi_coverage"),
        "pi_width": metrics.get("pi_width"),
        "models": [{"name": name, **base_runs.get(name, {})} for name in ensemble.model_names],
    }
