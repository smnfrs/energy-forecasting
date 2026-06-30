"""Ensemble methods for price models (§5c.5).

Nine methods total — seven weight-based plus two stacking. All are fitted on
out-of-fold (OOF) base-model predictions and evaluated on a held-out window
of base-model predictions. ``compare_ensemble_methods`` runs the full bake-off
and ``select_best_ensemble`` picks the holdout winner.

Each fitted ensemble exposes :py:meth:`predict(preds_matrix)` where
``preds_matrix`` is ``(n_samples, n_models)`` — the columns must come from the
same models, in the same order, that the ensemble was trained on. This is the
same shape ``modeling.gen_load.ensemble_gen_load`` already uses for stacking.

The legacy stubs ``modeling/blend.py`` and ``modeling/stacking.py`` are
removed; ensemble logic lives only here per the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge

from energy_forecasting.config.modeling import ENSEMBLE_METHODS
from energy_forecasting.modeling.metrics import calculate_metrics

# ── Container types ───────────────────────────────────────────────


@dataclass
class WeightEnsemble:
    """Linear combination of base-model predictions.

    Weights sum to 1 and are non-negative (where applicable). At predict time
    the same column ordering must be used.
    """

    method: str
    weights: np.ndarray  # (n_models,)
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


# ── Shared helpers ────────────────────────────────────────────────


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
    """Project to the probability simplex (clip to ≥0, sum to 1)."""
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Weight-based methods ──────────────────────────────────────────


def fit_simple_average(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
) -> WeightEnsemble:
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
    """``w_i ∝ 1 / (MAE_i + eps)``.

    Matches EP's production `_compute_inverse_mae_weights` in
    `src/modeling/blend.py:103`. EP fits these weights on per-model holdout
    MAE (not OOF), so this function should be passed the *holdout* window
    as ``preds_oof``/``y_oof`` for parity. See compare_ensemble_methods,
    which routes weight-based methods through the holdout window.
    """
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
    """``w_i ∝ 1 / (RMSE_i + eps)``.

    From EP's `scripts/compare_blend_strategies.py:59`.
    """
    y_true = y_oof.to_numpy(dtype=float)
    rmses = np.array(
        [
            float(np.sqrt(np.mean((y_true - preds_oof[m].to_numpy(float)) ** 2)))
            for m in preds_oof.columns
        ]
    )
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
    """Drop worst models, inverse-MAE on the remainder.

    From EP's `scripts/compare_blend_strategies.py:88`. ``keep`` defaults to
    ``min(6, n_models)`` matching EP's default.
    """
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


def fit_slsqp(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
) -> WeightEnsemble:
    """Constrained MAE minimisation: ``Σw=1, w≥0``."""
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
        metadata={"converged": bool(res.success), "final_mae": float(res.fun)},
    )


def fit_greedy_forward(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    max_models: int | None = None,
    diversity_weight: float = 0.0,
) -> WeightEnsemble:
    """Greedy ensemble (Caruana et al., 2004).

    Maintains a running ensemble as a count vector; each step adds the model
    whose inclusion most reduces validation MAE. ``diversity_weight`` adds a
    small bonus for models with low correlation to the current blend
    (``diversity_regularized`` uses ``α=0.05``).
    """
    P = preds_oof.to_numpy(dtype=float)
    y = y_oof.to_numpy(dtype=float)
    n_samples, n_models = P.shape
    if max_models is None:
        max_models = 50 * n_models  # generous cap, ~50 picks per model

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
                # Reward low-correlation (high-diversity) picks.
                mae -= diversity_weight * (1.0 - abs(corr))
            cand_maes[j] = mae
        best_j = int(np.argmin(cand_maes))
        counts[best_j] += 1
        running_sum += P[:, best_j]
        if cand_maes[best_j] >= best_mae - 1e-6:
            break
        best_mae = float(cand_maes[best_j])

    weights = counts.astype(float)
    return WeightEnsemble(
        method="diversity_regularized" if diversity_weight > 0 else "greedy_forward",
        weights=_normalise(weights),
        model_names=list(preds_oof.columns),
        metadata={"counts": counts.tolist()},
    )


def fit_hill_climbing(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    max_iter: int = 200,
) -> WeightEnsemble:
    """Start from the greedy solution, then accept any single-step
    swap/drop/add that lowers MAE."""
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
        metadata={"counts": counts.tolist(), "final_mae": best},
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
    """Random-restart hill climber with temperature schedule."""
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
        metadata={"counts": best_counts.tolist(), "final_mae": best},
    )


def fit_diversity_regularized(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    *,
    diversity_weight: float = 0.05,
) -> WeightEnsemble:
    return fit_greedy_forward(
        preds_oof,
        y_oof,
        diversity_weight=diversity_weight,
    )


# ── Stacking methods ──────────────────────────────────────────────


def fit_stacking_ridge(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
) -> StackEnsemble:
    meta = Ridge(positive=True, alpha=1.0)
    meta.fit(preds_oof.to_numpy(float), y_oof.to_numpy(float))
    return StackEnsemble(
        method="stacking_ridge",
        meta_learner=meta,
        model_names=list(preds_oof.columns),
    )


def fit_stacking_lgbm(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
) -> StackEnsemble:
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


# ── Method registry ───────────────────────────────────────────────


METHOD_FACTORIES: dict[str, Callable[[pd.DataFrame, pd.Series], Ensemble]] = {
    "simple_average": fit_simple_average,
    "inverse_mae": fit_inverse_mae,
    "inverse_rmse": fit_inverse_rmse,
    "top_k_trimmed": fit_top_k_trimmed,
    "slsqp_optimized": fit_slsqp,
    "greedy_forward": fit_greedy_forward,
    "hill_climbing": fit_hill_climbing,
    "simulated_annealing": fit_simulated_annealing,
    "diversity_regularized": fit_diversity_regularized,
    "stacking_ridge": fit_stacking_ridge,
    "stacking_lgbm": fit_stacking_lgbm,
}

# Weight-based methods are fitted on the *holdout* window — matches EP's
# production `_compute_inverse_mae_weights` in src/modeling/blend.py.
# Stacking methods are fitted on OOF (proper meta-learning, no double use
# of the holdout for both fit and evaluation).
HOLDOUT_FIT_METHODS = frozenset(
    {
        "simple_average",
        "inverse_mae",
        "inverse_rmse",
        "top_k_trimmed",
        "slsqp_optimized",
        "greedy_forward",
        "hill_climbing",
        "simulated_annealing",
        "diversity_regularized",
    }
)
OOF_FIT_METHODS = frozenset({"stacking_ridge", "stacking_lgbm"})


def fit_ensemble(method: str, preds: pd.DataFrame, y: pd.Series) -> Ensemble:
    if method not in METHOD_FACTORIES:
        raise ValueError(
            f"Unknown ensemble method {method!r}. Available: {sorted(METHOD_FACTORIES)}"
        )
    return METHOD_FACTORIES[method](preds, y)


# ── Compare & select ─────────────────────────────────────────────


def compare_ensemble_methods(
    preds_oof: pd.DataFrame,
    y_oof: pd.Series,
    preds_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    methods: Iterable[str] | None = None,
    include_base_models: bool = True,
) -> pd.DataFrame:
    """Fit and evaluate each method on the holdout window.

    Weight-based methods (``HOLDOUT_FIT_METHODS``) fit weights on
    ``(preds_holdout, y_holdout)`` and report MAE on the same window — this
    matches EP's `train_and_blend` in `src/modeling/blend.py:391-510`. The
    holdout MAE is in-sample for the weights but the base-model
    predictions are honest (each base model was trained on pool only).

    Stacking methods (``OOF_FIT_METHODS``) fit the meta-learner on
    ``(preds_oof, y_oof)`` and report MAE on the held-out window — proper
    out-of-sample evaluation for stacking.

    When ``include_base_models`` is true, each base model is also added to
    the comparison as a one-hot ``WeightEnsemble`` so the selection step
    never returns an ensemble that's worse than the best single model.
    """
    methods = list(methods) if methods is not None else list(ENSEMBLE_METHODS)
    if list(preds_oof.columns) != list(preds_holdout.columns):
        raise ValueError("preds_oof and preds_holdout must share columns in the same order.")

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Ensemble] = {}
    for method in methods:
        if method in HOLDOUT_FIT_METHODS:
            fit_X, fit_y = preds_holdout, y_holdout
        elif method in OOF_FIT_METHODS:
            fit_X, fit_y = preds_oof, y_oof
        else:
            logger.warning(
                f"Method {method!r} not in HOLDOUT_FIT_METHODS or OOF_FIT_METHODS; "
                f"defaulting to OOF-fit."
            )
            fit_X, fit_y = preds_oof, y_oof

        try:
            ens = fit_ensemble(method, fit_X, fit_y)
        except Exception as exc:  # noqa: BLE001 — surface any method failure
            logger.warning(f"Ensemble {method!r} failed during fit: {exc}")
            continue
        fitted[method] = ens
        preds = ens.predict(preds_holdout)
        metrics = calculate_metrics(y_holdout, preds)
        rows.append({"method": method, **metrics})

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
            rows.append({"method": method_label, **metrics})

    df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    df.attrs["fitted"] = fitted
    return df


def select_best_ensemble(
    comparison: pd.DataFrame,
) -> tuple[str, Ensemble, dict[str, float]]:
    """Pick the method with the lowest holdout MAE.

    Returns ``(method_name, fitted_ensemble, metrics_dict)``. The fitted
    objects are carried in ``comparison.attrs["fitted"]`` from
    :func:`compare_ensemble_methods`.
    """
    if comparison.empty:
        raise ValueError("No ensemble candidates to choose from.")
    fitted: dict[str, Ensemble] = comparison.attrs.get("fitted", {})
    if not fitted:
        raise ValueError(
            "comparison.attrs['fitted'] missing. Call compare_ensemble_methods first."
        )
    winner_row = comparison.iloc[0]
    method = str(winner_row["method"])
    ensemble = fitted[method]
    metrics = {k: float(v) for k, v in winner_row.items() if k != "method"}
    return method, ensemble, metrics


def ensemble_config_dict(
    ensemble: Ensemble,
    *,
    base_runs: dict[str, dict[str, Any]],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Build the JSON-serialisable summary written to ``ensemble_config.json``.

    ``base_runs`` maps ``model_name`` → ``{run_id, dataset_path,
    feature_version, hyperparams, preprocessing}`` so the file is sufficient
    to retrain the ensemble from scratch.

    ``metrics`` may contain ``conformal_quantile``/``pi_coverage``/
    ``pi_width`` (added by ``run_price_pipeline`` after post-hoc conformal
    calibration). When present they are surfaced at the top level so
    inference code can read them without parsing nested metrics.
    """
    if isinstance(ensemble, WeightEnsemble):
        ensemble_section: dict[str, Any] = {
            "method": ensemble.method,
            "weights": {
                name: float(w)
                for name, w in zip(ensemble.model_names, ensemble.weights, strict=True)
            },
        }
    else:
        ensemble_section = {
            "method": ensemble.method,
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
