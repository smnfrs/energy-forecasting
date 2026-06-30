"""Periodic retrain pipeline.

Price retrain:
  - Refit all non-zero-weight ensemble base models using stored hyperparams
    (no Optuna re-search; fast enough for GitHub Actions ~30-90 min)
  - Recompute SLSQP ensemble weights on fresh OOF predictions
  - Degradation check: if new MAE / old MAE > 1.20, flag needs_reselection
  - If no degradation: update ensemble_config.json and export new models to disk

Gen/load retrain:
  - 8-12 hours for all 218-fold sliding CV runs — exceeds GitHub Actions limit
  - Must be run manually on the tower via detached process per CLAUDE.md §Long-Running
  - Called from `make retrain-gen-load` which sets up the detached shell
  - This module provides `run_gen_load_retrain()` for programmatic use

Usage:
    energy-forecasting deploy retrain                 # price + gen/load config update
    energy-forecasting deploy retrain --price-only    # price only (CI-safe)
"""

from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from energy_forecasting.config import MLFLOW_TRACKING_URI
from energy_forecasting.config.modeling import BLEND_DEGRADATION_THRESHOLD, HOLDOUT_DAYS
from energy_forecasting.deploy.model_store import (
    ENSEMBLE_CONFIG_PATH,
    export_price_models,
    load_ensemble_config,
    production_model_names,
)

PRICE_RETRAIN_FEATURE_VERSIONS = ["fs_shap_top90", "fs_rfecv_optimum", "fs_shap_top247"]


def _apply_ema_overlay_for_retrain() -> None:
    """Re-apply EMA overlay to merged dataset before retraining.

    Uses only historical_forecasts already on disk (no live inference).
    """
    from energy_forecasting.config import PROCESSED_DATA_DIR
    from energy_forecasting.modeling.price import _overlay_ema_forecasts

    merged_path = PROCESSED_DATA_DIR / "merged.parquet"
    if not merged_path.exists():
        logger.warning(f"merged.parquet not found at {merged_path}, skipping EMA overlay")
        return

    df = pd.read_parquet(merged_path)
    df = _overlay_ema_forecasts(df)
    df.to_parquet(merged_path)
    logger.info(f"EMA overlay applied for retrain ({len(df)} rows)")


def _retrain_one_price_model(entry: dict) -> str | None:
    """Retrain one price base model using stored hyperparams. Returns new run_id."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from energy_forecasting.modeling.cv import TimeSeriesSplitter
    from energy_forecasting.modeling.datasets import find_dataset
    from energy_forecasting.modeling.training import train_model
    from energy_forecasting.modeling.tuning import _make_model as _make_price_model

    feature_version = entry["feature_version"]
    config = entry["config"]
    ds_name = f"price_{feature_version}"
    ds_path = find_dataset(ds_name)

    if ds_path is None:
        logger.warning(f"Dataset '{ds_name}' not found; skipping {entry['name']}")
        return None

    model = _make_price_model(config["model_type"], config["model_params"])
    tags = {
        "stage": "model_training",
        "feature_version": feature_version,
        "selection_step": "retrain",
    }
    cv = TimeSeriesSplitter(n_splits=5, mode="expanding")
    run_id = train_model(
        dataset_path=ds_path,
        model=model,
        experiment="price_production",
        tags=tags,
        scaler=config["scaler"],
        target_transform=config.get("target_transform", "none"),
        weight_half_life=config.get("weight_half_life"),
        cv=cv,
        holdout_days=HOLDOUT_DAYS,
        collect_oof=True,
    )
    logger.info(f"Retrained {entry['name']}: {run_id[:8]}")
    return run_id


def run_price_retrain(
    force: bool = False,
    holdout_days: int | None = None,
) -> dict:
    """Retrain all production price models and recompute ensemble weights.

    Steps:
    1. Apply EMA overlay to merged dataset
    2. Retrain each non-zero-weight base model using stored hyperparams
    3. Recompute SLSQP ensemble weights from fresh OOF predictions
    4. Degradation check
    5. If OK: update ensemble_config.json + export models

    Returns dict with:
        new_mae, old_mae, needs_reselection, config_path
    """
    old_config = load_ensemble_config()
    old_mae = old_config.get("metrics", {}).get("mae", float("inf"))
    prod_names = set(production_model_names(old_config))

    # 1. EMA overlay
    _apply_ema_overlay_for_retrain()

    # 2. Retrain each production model
    new_run_ids: dict[str, str] = {}
    for entry in old_config["models"]:
        if entry["name"] not in prod_names:
            continue
        try:
            run_id = _retrain_one_price_model(entry)
            if run_id:
                new_run_ids[entry["name"]] = run_id
        except Exception:
            logger.exception(f"Failed retrain for {entry['name']}")

    if not new_run_ids:
        raise RuntimeError("All price model retrains failed")

    # 3. Recompute ensemble weights
    new_config, new_mae = _recompute_ensemble(old_config, new_run_ids, holdout_days)

    # 4. Degradation check
    if old_mae > 0:
        ratio = new_mae / old_mae
        needs_reselection = ratio > (1.0 + BLEND_DEGRADATION_THRESHOLD)
    else:
        needs_reselection = False

    if needs_reselection and not force:
        logger.warning(
            f"Retrain degraded: new MAE={new_mae:.3f} vs old={old_mae:.3f} "
            f"(ratio={new_mae / old_mae:.2%} > threshold {1 + BLEND_DEGRADATION_THRESHOLD:.0%}). "
            "Not updating config. Pass --force to override."
        )
        return {
            "new_mae": new_mae,
            "old_mae": old_mae,
            "needs_reselection": True,
            "config_path": None,
        }

    # 5. Update config and export
    new_config["needs_reselection"] = needs_reselection
    ENSEMBLE_CONFIG_PATH.write_text(json.dumps(new_config, indent=2))
    logger.info(f"ensemble_config.json updated: MAE {old_mae:.3f} → {new_mae:.3f}")

    export_price_models(new_config)

    return {
        "new_mae": new_mae,
        "old_mae": old_mae,
        "needs_reselection": needs_reselection,
        "config_path": str(ENSEMBLE_CONFIG_PATH),
    }


def _recompute_ensemble(
    old_config: dict,
    new_run_ids: dict[str, str],
    holdout_days: int | None = None,
) -> tuple[dict, float]:
    """Load OOF + holdout predictions for the new runs and refit SLSQP weights."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from energy_forecasting.modeling.ensemble import (
        compare_ensemble_methods,
        ensemble_config_dict,
        select_best_ensemble,
    )
    from energy_forecasting.modeling.price import _ModelRun, _stack_predictions

    # Build updated model run list
    new_model_runs = []
    for entry in old_config["models"]:
        name = entry["name"]
        run_id = new_run_ids.get(name, entry["run_id"])  # fall back to old if not retrained
        new_model_runs.append(
            _ModelRun(
                name=name,
                run_id=run_id,
                model_type=entry["model_type"],
                feature_version=entry["feature_version"],
                config=entry["config"],
            )
        )

    # Stack predictions
    try:
        preds_oof, y_oof, preds_holdout, y_holdout = _stack_predictions(new_model_runs)
    except Exception:
        logger.exception("Failed to stack predictions for ensemble recompute")
        raise

    # Bake-off
    results = compare_ensemble_methods(preds_oof, y_oof, preds_holdout, y_holdout)
    best_method, best_weights = select_best_ensemble(results, preds_holdout, y_holdout)

    # New ensemble config
    new_config = ensemble_config_dict(
        method=best_method,
        weights=best_weights,
        model_runs=new_model_runs,
        preds_oof=preds_oof,
        y_oof=y_oof,
        preds_holdout=preds_holdout,
        y_holdout=y_holdout,
    )
    new_mae = float(new_config.get("metrics", {}).get("mae", float("inf")))
    return new_config, new_mae


def run_gen_load_retrain() -> None:
    """Retrain all gen/load models using stored hyperparams.

    WARNING: This takes 8-12 hours. Always run detached:
        setsid nohup energy-forecasting deploy retrain > logs/retrain.log 2>&1 &
        disown

    After completion, run export-models and upload new models to GitHub Release.
    """
    from energy_forecasting.config.modeling import GEN_LOAD_TARGETS
    from energy_forecasting.deploy.model_store import (
        GEN_LOAD_CONFIG_PATH,
        load_gen_load_config,
    )
    from energy_forecasting.modeling.gen_load import retrain_gen_load_from_existing

    gl_config = load_gen_load_config()

    for target, info in GEN_LOAD_TARGETS.items():
        for region in info["regions"]:
            combo_key = f"{target}/{region}"
            if combo_key not in gl_config["combos"]:
                logger.warning(f"No config for {combo_key}, skipping")
                continue
            entry = gl_config["combos"][combo_key]
            model_type = entry.get("model_type", "LGBMRegressor")
            logger.info(f"Retraining {combo_key} ({model_type})")
            try:
                new_run_id = retrain_gen_load_from_existing(target, region, model_type)
                gl_config["combos"][combo_key]["run_id"] = new_run_id
                GEN_LOAD_CONFIG_PATH.write_text(json.dumps(gl_config, indent=2))
                logger.info(f"  → {new_run_id[:8]}")
            except Exception:
                logger.exception(f"Failed retrain for {combo_key}")


def run_retrain(
    price_only: bool = False,
    force: bool = False,
    holdout_days: int | None = None,
) -> dict:
    """Top-level retrain entry point.

    price_only=True is the default for GitHub Actions (gen/load retrain is manual).
    """
    result = run_price_retrain(force=force, holdout_days=holdout_days)

    if not price_only:
        logger.info(
            "Gen/load retrain takes 8-12 hours and must be run detached on the tower. "
            "Skipping in this run. Use 'make retrain-gen-load' for a detached run."
        )

    return result
