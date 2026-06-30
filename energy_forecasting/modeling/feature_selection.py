"""Feature-selection pipeline for price models (§5c.2).

Produces *multiple* candidate feature sets rather than a single winner — the
downstream tuning step (§5c.3) and ensemble step (§5c.4–5c.5) compare them.
Each candidate is logged as a separate MLflow run in
``price/feature_selection`` so the chain back from the production ensemble to
the selection rationale is queryable.

Pipeline stages
---------------
1. **Correlation filter** — drop features with near-zero target correlation
   and one of each near-duplicate pair. Cheap, narrows the space before SHAP.
2. **SHAP importance** — LightGBM TreeExplainer on the filtered set, ranks
   features by mean |SHAP value| on the validation fold.
3. **SHAP top-N cutoff search** — sweep top-N from the ranking, retrain
   LightGBM at each N, pick the *local minima* of the CV-MAE curve. The
   plan calls for multiple candidates: small sets favour trees, larger sets
   give linear models room for regularisation to drive selection.
4. **RFECV** — sklearn's recursive feature elimination with the same
   TimeSeriesSplitter used elsewhere. Produces its own optimum plus the
   full curve so we can recover its local minima too.
5. **Log candidates** — write FULL, MAX, SHAP-derived sets, RFECV optimum
   to ``price/feature_selection`` runs with metrics from a reference
   LightGBM model. The chosen sets become the *dataset* inputs the tuning
   step iterates over.

This module never trains a final production model. It surfaces options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from energy_forecasting.config.modeling import (
    HOLDOUT_DAYS,
    SEARCH_CV_FOLDS,
)
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import load_dataset
from energy_forecasting.modeling.metrics import calculate_metrics
from energy_forecasting.modeling.mlflow_utils import TrackedRun

# ── Reference model ───────────────────────────────────────────────


def _reference_lgbm():
    """LightGBM regressor used as the selection reference.

    Deliberately conservative defaults — the goal is consistent rankings
    across feature sets, not state-of-the-art accuracy.
    """
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        n_jobs=-1,
    )


# ── Correlation filter ───────────────────────────────────────────


def correlation_filter(
    X: pd.DataFrame,
    y: pd.Series,
    min_target_corr: float = 0.02,
    max_pair_corr: float = 0.9999,
) -> list[str]:
    """Drop near-zero-target-correlation columns and one of each
    near-duplicate pair.

    Returns the surviving column names in the original order.
    """
    work = X.copy()
    # Target correlation — Spearman is robust to monotonic transforms.
    target_corr = work.apply(lambda col: col.corr(y, method="spearman"))
    keep_target_mask = target_corr.abs() >= min_target_corr
    dropped_low = target_corr.index[~keep_target_mask].tolist()
    work = work.loc[:, keep_target_mask]

    # Pairwise correlation — drop the second member of any pair above threshold.
    corr_matrix = work.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = {col for col in upper.columns if (upper[col] > max_pair_corr).any()}

    logger.info(
        f"correlation_filter: dropped {len(dropped_low)} for low target_corr, "
        f"{len(to_drop)} for high pair_corr ({len(work.columns) - len(to_drop)} kept)"
    )
    return [c for c in work.columns if c not in to_drop]


# ── SHAP importance ─────────────────────────────────────────────


def shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP value| per feature, sorted descending.

    Uses ``shap.TreeExplainer`` — appropriate for LightGBM/XGBoost/CatBoost.
    Lazy import so the rest of the module is usable without ``shap``.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP-based selection. "
            "Install with: pip install shap"
        ) from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


# ── SHAP cutoff search (local minima) ─────────────────────────────


@dataclass
class _CutoffResult:
    n_features: int
    mae: float
    rmse: float


def _evaluate_top_n(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
) -> _CutoffResult:
    model = _reference_lgbm()
    model.fit(X_train[feature_names], y_train)
    preds = model.predict(X_val[feature_names])
    metrics = calculate_metrics(y_val, preds)
    return _CutoffResult(len(feature_names), metrics["mae"], metrics["rmse"])


def shap_cutoff_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    shap_ranking: pd.Series,
    coarse_grid: tuple[int, ...] = (10, 20, 30, 40, 60, 90, 130, 180, 250),
    fine_window: int = 15,
) -> tuple[list[_CutoffResult], list[int]]:
    """Sweep top-N from the SHAP ranking. Returns full curve + local minima.

    Two passes: a coarse grid, then a fine grid (step=3) centred on each
    coarse local minimum. Each candidate evaluation is sub-second on a
    LightGBM reference model, so we can sweep generously — EP's optima
    have historically lived around n=30-50 which the previous default
    coarse grid (starting at 20) was missing in its fine-search radius.
    """
    valid_grid = [n for n in coarse_grid if n <= len(shap_ranking)]
    results: dict[int, _CutoffResult] = {}

    for n in valid_grid:
        feats = shap_ranking.index[:n].tolist()
        results[n] = _evaluate_top_n(X_train, y_train, X_val, y_val, feats)

    # Find local minima (strict) on the coarse curve, then refine each.
    coarse_curve = sorted(results.values(), key=lambda r: r.n_features)
    coarse_minima_n: list[int] = []
    for i, r in enumerate(coarse_curve):
        left = coarse_curve[i - 1].mae if i > 0 else float("inf")
        right = coarse_curve[i + 1].mae if i < len(coarse_curve) - 1 else float("inf")
        if r.mae <= left and r.mae <= right:
            coarse_minima_n.append(r.n_features)

    for centre in coarse_minima_n:
        for n in range(max(5, centre - fine_window), centre + fine_window + 1, 3):
            if n in results or n > len(shap_ranking):
                continue
            feats = shap_ranking.index[:n].tolist()
            results[n] = _evaluate_top_n(X_train, y_train, X_val, y_val, feats)

    curve = sorted(results.values(), key=lambda r: r.n_features)
    local_minima: list[int] = []
    for i, r in enumerate(curve):
        left = curve[i - 1].mae if i > 0 else float("inf")
        right = curve[i + 1].mae if i < len(curve) - 1 else float("inf")
        if r.mae < left and r.mae <= right:
            local_minima.append(r.n_features)

    if not local_minima and curve:
        local_minima.append(min(curve, key=lambda r: r.mae).n_features)

    return curve, local_minima


# ── RFECV selection ──────────────────────────────────────────────


def rfecv_select(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splitter: TimeSeriesSplitter,
    step: int = 5,
    min_features: int = 10,
) -> tuple[list[str], dict[int, float]]:
    """sklearn RFECV with the project's TimeSeriesSplitter.

    Returns (selected feature names, n_features → mean CV MAE curve).
    Uses negative MAE so sklearn's "higher is better" convention works.
    """
    from sklearn.feature_selection import RFECV

    rfecv = RFECV(
        estimator=_reference_lgbm(),
        step=step,
        min_features_to_select=min_features,
        cv=list(cv_splitter.split(X.index)),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    chosen = X.columns[rfecv.support_].tolist()

    # Curve: in sklearn >= 1.3 cv_results_["mean_test_score"] is per n_features.
    cv_results = rfecv.cv_results_
    n_axis = cv_results.get("n_features", None)
    if n_axis is None:
        # Older sklearn: scores indexed by step.
        n_axis = np.arange(
            min_features,
            min_features + len(cv_results["mean_test_score"]) * step,
            step,
        )
    curve = {
        int(n): -float(score)
        for n, score in zip(n_axis, cv_results["mean_test_score"], strict=False)
    }
    return chosen, curve


# ── Candidate logging helpers ─────────────────────────────────────


def _evaluate_set_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    cv_splitter: TimeSeriesSplitter,
) -> dict[str, float]:
    """Reference LightGBM mean CV metrics over the configured splitter."""
    fold_metrics: list[dict[str, float]] = []
    for train_idx, test_idx in cv_splitter.split(X.index):
        model = _reference_lgbm()
        model.fit(X.iloc[train_idx][feature_names], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx][feature_names])
        fold_metrics.append(calculate_metrics(y.iloc[test_idx], preds))
    return {
        f"cv_{k}": float(np.mean([m[k] for m in fold_metrics]))
        for k in fold_metrics[0]
    }


def _log_candidate(
    name: str,
    feature_names: list[str],
    metrics: dict[str, float],
    dataset_path: Path,
    extra_tags: dict[str, str] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> str:
    """Log a candidate feature set as an MLflow run."""
    tags = {
        "stage": "feature_selection",
        "feature_version": name,
        "holdout_days": str(HOLDOUT_DAYS),
        "cv_folds": str(SEARCH_CV_FOLDS),
        "cv_mode": "expanding",
        "target_transform": "none",
        "candidate_kind": (extra_tags or {}).get("candidate_kind", "manual"),
    }
    if extra_tags:
        tags.update(extra_tags)

    with TrackedRun("price_feature_selection", **tags) as run:
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("source_dataset", dataset_path.stem)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"features": feature_names}, "feature_list.json")
        if artifacts:
            for artifact_name, artifact_data in artifacts.items():
                mlflow.log_dict(artifact_data, artifact_name)
        return run.info.run_id


# ── Orchestrator ─────────────────────────────────────────────────


def run_feature_selection(
    dataset_path: Path,
    *,
    holdout_days: int = HOLDOUT_DAYS,
    cv_folds: int = SEARCH_CV_FOLDS,
    cv_mode: str = "expanding",
    extra_candidates: dict[str, list[str]] | None = None,
    use_shap: bool = True,
    use_rfecv: bool = True,
    rfecv_input_size: int = 80,
) -> dict[str, dict[str, Any]]:
    """Run correlation → SHAP → RFECV and log every candidate to MLflow.

    The three filters are sequential, not parallel: correlation drops
    obviously redundant features, SHAP ranks the survivors by importance,
    and RFECV is then asked to *recursively narrow* the SHAP top-N. The
    earlier implementation passed the full correlation-filtered set (often
    ~260 features) to RFECV, which made it 5-10× slower for little benefit
    — the same features SHAP would have dropped anyway took most of the
    elimination steps.

    Parameters
    ----------
    dataset_path
        Parquet from ``prepare_dataset`` for ``PRICE_FEATURES_MAX``.
    holdout_days, cv_folds, cv_mode
        Evaluation harness for ranking candidates against each other.
    extra_candidates
        Optional pre-curated lists to log alongside the discovered ones
        (e.g. ``{"slim": PRICE_FEATURES_SLIM, "full": PRICE_FEATURES_FULL}``).
        Each is filtered down to the columns actually present in the dataset.
    use_shap, use_rfecv
        Toggle the expensive selectors. Both default on; useful as kill
        switches when debugging.
    rfecv_input_size
        How many SHAP-top features RFECV starts from. RFECV will narrow
        further from there. When ``use_shap`` is false the value is
        ignored and RFECV falls back to the correlation-filtered set.

    Returns
    -------
    dict
        Per-candidate ``{name: {features, mae, run_id, kind}}``.
    """
    X, y = load_dataset(dataset_path)

    pool_idx, holdout_idx = carve_holdout(X.index, holdout_days)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_holdout, y_holdout = X.iloc[holdout_idx], y.iloc[holdout_idx]

    cv_splitter = TimeSeriesSplitter(n_splits=cv_folds, mode=cv_mode)

    candidates: dict[str, dict[str, Any]] = {}

    def _record(
        name: str,
        features: list[str],
        kind: str,
        artifacts: dict[str, Any] | None = None,
    ):
        metrics = _evaluate_set_with_cv(X_pool, y_pool, features, cv_splitter)
        # Also report holdout metrics as a sanity check.
        model = _reference_lgbm()
        model.fit(X_pool[features], y_pool)
        holdout_preds = model.predict(X_holdout[features])
        for k, v in calculate_metrics(y_holdout, holdout_preds).items():
            metrics[f"holdout_{k}"] = v

        run_id = _log_candidate(
            name=name,
            feature_names=features,
            metrics=metrics,
            dataset_path=dataset_path,
            extra_tags={"candidate_kind": kind},
            artifacts=artifacts,
        )
        candidates[name] = {
            "features": features,
            "metrics": metrics,
            "run_id": run_id,
            "kind": kind,
        }
        logger.info(
            f"[{kind}] {name}: n={len(features)} "
            f"cv_mae={metrics['cv_mae']:.3f} holdout_mae={metrics['holdout_mae']:.3f}"
        )

    # 0. Always record the unfiltered MAX baseline.
    _record("max", X.columns.tolist(), kind="baseline")

    # 1. Correlation-filtered baseline.
    filtered = correlation_filter(X_pool, y_pool)
    _record("corr_filtered", filtered, kind="correlation")

    # 2. SHAP cutoffs (optional — requires shap).
    shap_ranking: pd.Series | None = None
    if use_shap:
        ref = _reference_lgbm()
        ref.fit(X_pool[filtered], y_pool)
        shap_ranking = shap_importance(ref, X_pool[filtered])

        # Use a small temporal sub-split inside the pool as the SHAP-cutoff
        # validation slice — the holdout stays untouched.
        cutoff = int(len(X_pool) * 0.85)
        X_tr, X_val = X_pool.iloc[:cutoff], X_pool.iloc[cutoff:]
        y_tr, y_val = y_pool.iloc[:cutoff], y_pool.iloc[cutoff:]

        curve, _minima = shap_cutoff_search(X_tr, y_tr, X_val, y_val, shap_ranking)
        # Record every evaluated SHAP point — the previous behaviour
        # silently dropped the coarse-grid points that weren't local minima,
        # so the small-n range never made it into downstream tuning candidate
        # selection. We let the ranking step pick top-K across the full curve.
        for r in curve:
            top_n = shap_ranking.index[: r.n_features].tolist()
            _record(f"shap_top{r.n_features}", top_n, kind="shap")

        # Save SHAP ranking and cutoff curve to disk for 5d notebooks, and
        # log them to a lightweight meta MLflow run so they're queryable.
        artifacts_dir = dataset_path.parent
        shap_rank_df = shap_ranking.reset_index()
        shap_rank_df.columns = ["feature", "importance"]
        shap_rank_df.to_parquet(artifacts_dir / "price_fs_shap_ranking.parquet", index=False)

        shap_curve_df = pd.DataFrame(
            [(r.n_features, r.mae, r.rmse) for r in curve],
            columns=["n_features", "mae", "rmse"],
        )
        shap_curve_df.to_parquet(artifacts_dir / "price_fs_shap_curve.parquet", index=False)
        logger.info(
            f"Saved SHAP ranking ({len(shap_rank_df)} features) and "
            f"cutoff curve ({len(shap_curve_df)} points) to {artifacts_dir}"
        )

        shap_rank_records = shap_rank_df.to_dict(orient="records")
        shap_curve_records = shap_curve_df.to_dict(orient="records")
        with TrackedRun(
            "price_feature_selection",
            dataset_name=dataset_path.stem,
            stage="feature_selection",
            feature_version="meta_shap",
            candidate_kind="meta",
            holdout_days=str(holdout_days),
            cv_folds=str(cv_folds),
            cv_mode=cv_mode,
            target_transform="none",
        ):
            mlflow.log_dict({"ranking": shap_rank_records}, "shap_ranking.json")
            mlflow.log_dict({"curve": shap_curve_records}, "shap_curve.json")

    # 3. RFECV (optional — the slowest step; protect the rest of the
    # pipeline so an RFECV failure doesn't kill the overnight run after
    # SHAP candidates have already been logged).
    if use_rfecv:
        if shap_ranking is not None:
            n_input = min(rfecv_input_size, len(shap_ranking))
            rfecv_input_features = shap_ranking.index[:n_input].tolist()
            logger.info(
                f"RFECV starting from SHAP top-{n_input} "
                f"(out of {len(filtered)} corr-filtered features)"
            )
        else:
            rfecv_input_features = filtered
            logger.info(
                f"RFECV starting from corr-filtered set "
                f"(n={len(filtered)}, no SHAP narrowing)"
            )
        try:
            rfecv_features, rfecv_curve = rfecv_select(
                X_pool[rfecv_input_features], y_pool, cv_splitter,
            )
            rfecv_curve_df = pd.DataFrame(
                sorted(rfecv_curve.items()), columns=["n_features", "mae"],
            )
            rfecv_curve_df.to_parquet(
                dataset_path.parent / "price_fs_rfecv_curve.parquet", index=False,
            )
            logger.info(f"Saved RFECV curve ({len(rfecv_curve_df)} points) to {dataset_path.parent}")
            _record(
                "rfecv_optimum",
                rfecv_features,
                kind="rfecv",
                artifacts={"rfecv_curve.json": rfecv_curve_df.to_dict(orient="records")},
            )
        except Exception as exc:  # noqa: BLE001 — keep overnight robust
            logger.error(f"RFECV failed: {exc}. Continuing without RFECV candidate.")

    # 4. User-supplied lists.
    if extra_candidates:
        for name, feats in extra_candidates.items():
            usable = [f for f in feats if f in X.columns]
            if not usable:
                logger.warning(f"Skipping extra candidate {name!r}: no columns present in dataset")
                continue
            _record(name, usable, kind="manual")

    return candidates
