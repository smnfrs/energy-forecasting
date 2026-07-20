"""End-to-end price-model training pipeline (§5c.4).

Orchestrates the chain:

    feature dataset prep  →  feature_selection (optional)  →  tuning
                                                          ↓
                  ensemble selection  ←  final retrain with OOF capture

The output is a JSON ensemble config under
``models/ensemble_config.json`` that records every base model's run_id,
hyperparameters, preprocessing, and weight in the winning ensemble.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from loguru import logger

from energy_forecasting.config import MODELS_DIR
from energy_forecasting.config.features import (
    PRICE_FEATURES_FULL,
    PRICE_FEATURES_MAX,
    PRICE_FEATURES_SLIM,
)
from energy_forecasting.config.modeling import (
    FEATURE_CONTRACT,
    HOLDOUT_DAYS,
    PRICE_HOLDOUT_MIN_OWN_FORECAST_FRACTION,
    SEARCH_CV_FOLDS,
    VALIDATION_CV_FOLDS,
)
from energy_forecasting.data.io import load_parquet
from energy_forecasting.features.forecast_coverage import assert_price_holdout_forecast_coverage
from energy_forecasting.features.forecast_inputs import build_forecast_columns
from energy_forecasting.features.validation import validate_price_feature_list
from energy_forecasting.modeling.cv import TimeSeriesSplitter
from energy_forecasting.modeling.datasets import find_dataset, prepare_dataset
from energy_forecasting.modeling.ensemble import (
    build_production_ensemble,
    ensemble_config_dict,
    fetch_prediction_artifacts,
    stack_model_predictions,
)
from energy_forecasting.modeling.mlflow_utils import TrackedRun, ensure_mlflow_tracking
from energy_forecasting.modeling.training import train_model
from energy_forecasting.modeling.tuning import (
    PRICE_LINEAR_TYPES,
    PRICE_TREE_TYPES,
    _make_model,
    tune_linear_model,
    tune_tree_model,
)

# Default target column on the merged dataset.
PRICE_TARGET = "target_price"


FEATURE_LISTS: dict[str, list[str]] = {
    "slim": PRICE_FEATURES_SLIM,
    "full": PRICE_FEATURES_FULL,
    "max": PRICE_FEATURES_MAX,
}


# ── Dataset prep ─────────────────────────────────────────────────



def prepare_price_dataset(
    feature_version: str,
    *,
    merged_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Compute (or look up) the price feature dataset for one feature list.

    Long-window rolling features (e.g. ``price_ewma_2160_d1``,
    ``price_d30_d1_avg``) leave NaN over the first ~90 days. Linear models
    refuse NaN, so we drop those rows here — keeping the on-disk dataset
    consistent with what every model type can consume.
    """
    if feature_version not in FEATURE_LISTS:
        raise ValueError(
            f"Unknown feature_version {feature_version!r}. Available: {sorted(FEATURE_LISTS)}"
        )

    dataset_name = f"price_{feature_version}"
    existing = find_dataset(dataset_name)
    if existing and not force:
        logger.info(f"Reusing existing dataset {existing}")
        return existing

    merged_path = merged_path or Path("data/processed/merged.parquet")
    feature_list = FEATURE_LISTS[feature_version]
    validate_price_feature_list(feature_list)
    df = load_parquet(merged_path)
    logger.info(f"Loaded merged data: {df.shape} from {merged_path}")
    df = build_forecast_columns(df)
    path = prepare_dataset(
        df,
        feature_list,
        target_col=PRICE_TARGET,
        name=dataset_name,
    )
    # Drop the warm-up window where rolling stats are still NaN.
    full = pd.read_parquet(path)
    cleaned = full.dropna()
    dropped = len(full) - len(cleaned)
    if dropped:
        logger.info(f"Dropped {dropped} warm-up rows with NaN features ({len(cleaned)} remain)")
        cleaned.to_parquet(path)
    return path




def _assert_dataset_holdout_forecast_coverage(
    dataset_path: Path,
    *,
    merged_path: Path | None = None,
) -> dict[str, Any]:
    """Guard against blending/calibrating on a fallback-sourced holdout."""
    merged_path = merged_path or Path("data/processed/merged.parquet")
    raw = load_parquet(merged_path)
    dataset_index = pd.read_parquet(dataset_path).index
    report = assert_price_holdout_forecast_coverage(
        pd.DatetimeIndex(dataset_index),
        raw,
        min_own_fraction=PRICE_HOLDOUT_MIN_OWN_FORECAST_FRACTION,
        context=f"{dataset_path.name} price holdout",
    )
    logger.info(
        f"Forecast source coverage for {dataset_path.name} holdout: "
        f"own={report['own_fraction'] * 100:.2f}% counts={report['counts']}"
    )
    return report


def prepare_subset_dataset(parent_path: Path, columns: list[str], name: str) -> Path:
    """Materialise a sub-dataset of an existing price dataset.

    Used by the feature-selection workflow: ``run_feature_selection`` returns
    a list of candidate feature subsets of the MAX dataset; each candidate
    is persisted here as its own Parquet so the tuning step can iterate
    over them like any other ``feature_version``.

    The target column (suffix ``__target``) is always preserved.
    """
    from energy_forecasting.modeling.datasets import DATASET_DIR, TARGET_COL_SUFFIX

    parent = pd.read_parquet(parent_path)
    target_cols = [c for c in parent.columns if c.endswith(TARGET_COL_SUFFIX)]
    if len(target_cols) != 1:
        raise ValueError(
            f"Parent dataset must have exactly one target column "
            f"(suffix '{TARGET_COL_SUFFIX}'); found {target_cols}"
        )
    missing = [c for c in columns if c not in parent.columns]
    if missing:
        raise KeyError(
            f"Subset asked for {len(missing)} columns not present in parent "
            f"dataset: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    keep = list(columns) + target_cols
    subset = parent[keep]
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / f"price_{name}.parquet"
    subset.to_parquet(out)
    logger.info(f"Subset dataset '{name}': {subset.shape} → {out}")
    return out


# ── Final retrain & prediction collection ────────────────────────


@dataclass
class _ModelRun:
    name: str
    run_id: str
    model_type: str
    feature_version: str
    config: dict[str, Any]


def _train_winner(
    dataset_path: Path,
    config: dict[str, Any],
    feature_version: str,
    cv_folds: int,
) -> str:
    """Retrain a tuning winner with VALIDATION_CV_FOLDS, collecting OOF preds."""
    model = _make_model(config["model_type"], config["model_params"])
    tags = {
        "stage": "model_training",
        "feature_version": feature_version,
        "feature_contract": FEATURE_CONTRACT,
        "holdout_days": str(HOLDOUT_DAYS),
        "cv_folds": str(cv_folds),
        "cv_mode": "expanding",
        "target_transform": config["target_transform"],
        "selection_step": "winner_retrain",
    }
    cv = TimeSeriesSplitter(n_splits=cv_folds, mode="expanding")
    return train_model(
        dataset_path=dataset_path,
        model=model,
        experiment="price_model_training",
        tags=tags,
        scaler=config["scaler"],
        target_transform=config["target_transform"],
        weight_half_life=config.get("weight_half_life"),
        cv=cv,
        holdout_days=HOLDOUT_DAYS,
        collect_oof=True,
    )


def _fetch_predictions(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OOF + holdout prediction parquets for one run."""
    return fetch_prediction_artifacts(run_id)


def _stack_predictions(
    model_runs: list[_ModelRun],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Assemble validated (preds_oof, y_oof, preds_holdout, y_holdout)."""
    preds_oof, y_oof, preds_hold, y_hold, _ = stack_model_predictions(
        model_runs,
        prediction_loader=_fetch_predictions,
    )
    return preds_oof, y_oof, preds_hold, y_hold


# ── Top-level orchestrator ───────────────────────────────────────


def run_price_pipeline(
    feature_versions: list[str],
    *,
    tree_types: tuple[str, ...] = PRICE_TREE_TYPES,
    linear_types: tuple[str, ...] = PRICE_LINEAR_TYPES,
    search_cv_folds: int = SEARCH_CV_FOLDS,
    validation_cv_folds: int = VALIDATION_CV_FOLDS,
    output_config: Path | None = None,
    use_feature_selection: bool = False,
    feature_selection_top_k: int = 4,
    feature_selection_use_rfecv: bool = False,
    precomputed_datasets: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Run the full price training pipeline.

    Parameters
    ----------
    feature_versions
        One or more entries from ``FEATURE_LISTS`` (``slim``/``full``/``max``).
        For each, the orchestrator tunes every model type and contributes
        one winner per (model_type, feature_version) to the ensemble pool.
        Ignored when ``use_feature_selection`` is true.
    tree_types, linear_types
        Model families to include. Pass a shorter tuple to skip families
        during quick smoke runs.
    search_cv_folds
        Folds used during grid search (fast).
    validation_cv_folds
        Folds used during the final retrain (more reliable OOF + holdout).
    output_config
        Where to write ``ensemble_config.json``. Defaults to
        ``MODELS_DIR/ensemble_config.json``.
    use_feature_selection
        If true: prep the MAX dataset, run ``run_feature_selection`` on it,
        then iterate tuning over the top-K candidate subsets discovered.
        Overrides ``feature_versions``.
    feature_selection_top_k
        Number of candidate feature sets to feed into tuning (sorted by the
        feature-selection holdout MAE on the reference LightGBM).
    feature_selection_use_rfecv
        Whether to include RFECV in feature selection. RFECV is the slowest
        step (~hour+ on MAX); off by default for overnight runs.
    precomputed_datasets
        Optional ``{feature_version: parquet_path}`` mapping of already-built
        datasets. When given, the pipeline skips both dataset prep and feature
        selection and tunes every model family on each supplied dataset. Used
        for controlled model-focused experiments that reuse existing
        feature-selected parquets. Overrides ``feature_versions`` and
        ``use_feature_selection``.
    """
    if not feature_versions and not use_feature_selection and not precomputed_datasets:
        raise ValueError("feature_versions must list at least one entry")

    ensure_mlflow_tracking()
    output_config = output_config or (MODELS_DIR / "ensemble_config.json")

    # 1. Datasets.
    datasets: dict[str, Path] = {}
    # Per-family feature_version mapping. Trees prefer the feature-selected
    # sets (slim + SHAP minima + RFECV) where the rolling rolling-stat
    # redundancy is already pruned; linear models get every candidate
    # including ``max`` so regularisation can sift through the lot.
    tree_fv_set: set[str] = set()
    linear_fv_set: set[str] = set()

    if precomputed_datasets:
        logger.info("=== Using precomputed datasets (skipping prep + feature selection) ===")
        for fv, path in precomputed_datasets.items():
            datasets[fv] = Path(path)
            tree_fv_set.add(fv)
            linear_fv_set.add(fv)
    elif use_feature_selection:
        from energy_forecasting.modeling.feature_selection import (
            run_feature_selection,
        )

        logger.info("=== Feature selection on PRICE_FEATURES_MAX ===")
        max_path = prepare_price_dataset("max")
        slim_path = prepare_price_dataset("slim")
        slim_features = pd.read_parquet(slim_path).columns.tolist()
        slim_features = [c for c in slim_features if not c.endswith("__target")]

        candidates = run_feature_selection(
            max_path,
            use_rfecv=feature_selection_use_rfecv,
            extra_candidates={"slim": slim_features},
        )
        # Rank candidates by reference-LightGBM holdout MAE.
        ranked = sorted(
            candidates.items(),
            key=lambda kv: kv[1]["metrics"].get("holdout_mae", float("inf")),
        )
        chosen = ranked[:feature_selection_top_k]
        logger.info(
            "Feature-selection winners: "
            + ", ".join(
                f"{name} (cv_mae={info['metrics'].get('cv_mae', float('nan')):.3f}, "
                f"holdout_mae={info['metrics'].get('holdout_mae', float('nan')):.3f})"
                for name, info in chosen
            )
        )
        # Trees and linear both tune on the feature-selected winners. The
        # unfiltered ``max`` set was dropped 2026-06-07 — it never earned
        # ensemble weight and inflated the linear grid.
        for cand_name, cand_info in chosen:
            fv = f"fs_{cand_name}"
            datasets[fv] = prepare_subset_dataset(
                max_path,
                cand_info["features"],
                name=fv,
            )
            tree_fv_set.add(fv)
            linear_fv_set.add(fv)
    else:
        for fv in feature_versions:
            datasets[fv] = prepare_price_dataset(fv)
            tree_fv_set.add(fv)
            linear_fv_set.add(fv)

    # Guard before tuning, blending, and conformal calibration. This uses the
    # exact final dataset index and the same holdout split train_model() will
    # later carve, so a coverage hole cannot silently contaminate weights or PI.
    for fv, ds_path in sorted(datasets.items()):
        logger.info(f"=== Forecast source guard: feature_version={fv} ===")
        _assert_dataset_holdout_forecast_coverage(ds_path)

    # 2. Tuning per (model_type, feature_version). Wrap each
    # (model_type, feature_version) in a try/except so one slow / fragile
    # combination doesn't abort the whole overnight run.
    winners: dict[str, dict[str, Any]] = {}  # name → tuning result
    for fv in sorted(datasets):
        ds_path = datasets[fv]
        applicable_trees = [mt for mt in tree_types if fv in tree_fv_set]
        applicable_linear = [mt for mt in linear_types if fv in linear_fv_set]
        if not applicable_trees and not applicable_linear:
            continue
        logger.info(
            f"=== Tuning on feature_version={fv} "
            f"(trees={applicable_trees}, linear={applicable_linear}) ==="
        )
        for mt in applicable_trees:
            try:
                cfg = tune_tree_model(
                    ds_path,
                    mt,
                    feature_version=fv,
                    cv_folds=search_cv_folds,
                )
                winners[f"{mt}__{fv}"] = cfg
            except Exception as exc:  # noqa: BLE001 — keep overnight robust
                logger.error(f"tune_tree_model({mt}, fv={fv}) failed: {exc}")
        for mt in applicable_linear:
            try:
                cfg = tune_linear_model(
                    ds_path,
                    mt,
                    feature_version=fv,
                    cv_folds=search_cv_folds,
                )
                winners[f"{mt}__{fv}"] = cfg
            except Exception as exc:  # noqa: BLE001 — keep overnight robust
                logger.error(f"tune_linear_model({mt}, fv={fv}) failed: {exc}")

    if not winners:
        raise RuntimeError("All tuning runs failed — no winners to retrain.")

    # 3a. Prune dominated configs before the expensive retrain step.
    # Within each model type, keep only feature versions whose cv_mae is
    # within 20% of that model type's best cv_mae. This avoids spending
    # VALIDATION_CV_FOLDS × full-dataset training time on configs unlikely to survive
    # final ensemble selection, without touching cross-family diversity
    # (linear vs tree families are never compared against each other here).
    from collections import defaultdict

    by_type: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, cfg in winners.items():
        by_type[cfg["model_type"]][name] = cfg

    pruned_winners: dict[str, Any] = {}
    for mt, mt_winners in by_type.items():
        best_cv = min(c["cv_mae"] for c in mt_winners.values())
        threshold = best_cv * 1.2
        for name, cfg in mt_winners.items():
            if cfg["cv_mae"] <= threshold:
                pruned_winners[name] = cfg
            else:
                logger.info(f"Pruning {name} (cv_mae={cfg['cv_mae']:.3f} > {threshold:.3f})")
    winners = pruned_winners

    # 3. Final retrain per winner — captures OOF + holdout preds. Same
    # defensive wrapping per winner so MAPIE/training failures on one
    # config don't lose all other models' work.
    model_runs: list[_ModelRun] = []
    for name, cfg in winners.items():
        fv = name.split("__")[-1]
        ds_path = datasets[fv]
        logger.info(f"=== Retraining winner {name} (cv_mae={cfg.get('cv_mae'):.3f}) ===")
        try:
            run_id = _train_winner(
                ds_path,
                cfg,
                feature_version=fv,
                cv_folds=validation_cv_folds,
            )
        except Exception as exc:  # noqa: BLE001 — keep overnight robust
            logger.error(f"_train_winner({name}) failed: {exc}")
            continue
        model_runs.append(
            _ModelRun(
                name=name,
                run_id=run_id,
                model_type=cfg["model_type"],
                feature_version=fv,
                config=cfg,
            )
        )

    if not model_runs:
        raise RuntimeError("All retrain runs failed — no models to ensemble.")

    # 4. Assemble validated predictions, select category-floored candidates,
    # fit EP-style inverse-MAE weights on the recent holdout, and calibrate
    # conformal intervals on those in-sample holdout residuals.
    production = build_production_ensemble(
        model_runs,
        prediction_loader=_fetch_predictions,
        strict_categories=True,
    )
    method = production.method
    ensemble = production.ensemble
    metrics = production.metrics

    logger.info(
        f"Ensemble winner: {method} (holdout MAE={metrics['mae']:.3f}, "
        f"RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}, "
        f"PI coverage={metrics['pi_coverage']:.2%} "
        f"@ q={metrics['conformal_quantile']:.3f})"
    )

    # 5. Log the production ensemble to MLflow. Method bakeoff comparison is
    # diagnostic-only and lives in scripts/ensemble_method_comparison.py.
    base_runs = {
        mr.name: {
            "run_id": mr.run_id,
            "model_type": mr.model_type,
            "feature_version": mr.feature_version,
            "config": mr.config,
        }
        for mr in model_runs
    }
    config_dict = ensemble_config_dict(ensemble, base_runs=base_runs, metrics=metrics)
    numeric_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    with TrackedRun(
        "price_production",
        dataset_name="ensemble_production",
        stage="production",
        ensemble_step="ensemble_production",
        feature_version="+".join(feature_versions)
        if feature_versions
        else "+".join(sorted(datasets)),
        feature_contract=FEATURE_CONTRACT,
        holdout_days=str(HOLDOUT_DAYS),
        cv_folds=str(validation_cv_folds),
        cv_mode="expanding",
        target_transform="mixed",
        model_class=type(ensemble).__name__,
    ):
        mlflow.log_param("ensemble_method", method)
        mlflow.log_param("n_base_models", len(model_runs))
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_dict(config_dict, "ensemble_config.json")
        mlflow.log_dict(
            production.candidate_metrics.to_dict(orient="records"),
            "ensemble_candidate_metrics.json",
        )
        mlflow.log_dict(production.alignment_metadata, "ensemble_alignment.json")

    # 6. Persist the config alongside the repo's other model artifacts.
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Wrote ensemble config to {output_config}")

    return {
        "winners": winners,
        "model_runs": production.selected_model_runs,
        "comparison": production.comparison,
        "ensemble_method": method,
        "metrics": metrics,
        "config_path": output_config,
    }
