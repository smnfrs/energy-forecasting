"""Periodic retrain pipeline.

Price retrain:
  - Default mode retrains the current positive-weight production members on
    fresh merged data, then recomputes EP-style inverse-MAE holdout weights.
  - Reselection mode retrains every configured candidate before rebuilding the
    category-floored EP production member set.
  - Degradation check: if new MAE / old MAE > 1.20, flag needs_reselection.
  - If no degradation: update ensemble_config.json and export new models to disk.

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
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from energy_forecasting.config import (
    DEPLOY_DATA_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from energy_forecasting.config.modeling import (
    ENSEMBLE_DEGRADATION_THRESHOLD,
    FEATURE_CONTRACT,
    HOLDOUT_DAYS,
)
from energy_forecasting.deploy.model_store import (
    ENSEMBLE_CONFIG_PATH,
    export_price_models,
    load_ensemble_config,
    production_model_names,
)

RETRAIN_HISTORY_PATH = DEPLOY_DATA_DIR / "retrain_history.json"
_RETRAIN_HISTORY_MAX = 90

PRICE_RETRAIN_FEATURE_VERSIONS = ["fs_shap_top90", "fs_rfecv_optimum", "fs_shap_top247"]


def _append_retrain_history(entry: dict) -> None:
    """Append one retrain event to deploy/data/retrain_history.json (rolling 90 events)."""
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RETRAIN_HISTORY_PATH.exists():
        try:
            history: list[dict] = json.loads(RETRAIN_HISTORY_PATH.read_text())
        except Exception:
            history = []
    else:
        history = []
    history.append(entry)
    history = history[-_RETRAIN_HISTORY_MAX:]
    RETRAIN_HISTORY_PATH.write_text(json.dumps(history, indent=2))
    logger.info(f"Retrain history updated ({len(history)} events)")



def _price_feature_columns(feature_version: str) -> list[str]:
    """Load the frozen production feature recipe for one price feature version."""
    from energy_forecasting.features.validation import validate_price_feature_list
    from energy_forecasting.modeling.datasets import DATASET_DIR, TARGET_COL_SUFFIX

    cols_path = MODELS_DIR / "price_feature_cols.json"
    if cols_path.exists():
        payload = json.loads(cols_path.read_text())
        if feature_version in payload:
            cols = [c for c in payload[feature_version] if c != "__index_level_0__"]
            validate_price_feature_list(cols)
            return cols

    # Bootstrap fallback for local/dev configs that predate price_feature_cols.json.
    ds_path = DATASET_DIR / f"price_{feature_version}.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"No frozen feature recipe found for {feature_version}. Expected {cols_path} "
            f"or existing {ds_path}."
        )
    try:
        import pyarrow.parquet as pq

        names = pq.read_schema(ds_path).names
    except Exception:
        import pandas as pd

        names = list(pd.read_parquet(ds_path).columns)
    cols = [c for c in names if not c.endswith(TARGET_COL_SUFFIX) and c != "__index_level_0__"]
    validate_price_feature_list(cols)
    return cols


def rebuild_price_dataset_from_merged(
    feature_version: str,
    *,
    merged_path=None,
) -> Path:
    """Rebuild one price dataset from the current merged parquet and frozen columns."""
    import pandas as pd

    from energy_forecasting.config.features import PRICE_FEATURES_MAX
    from energy_forecasting.features.engine import engineer_features
    from energy_forecasting.features.forecast_inputs import build_forecast_columns
    from energy_forecasting.modeling.datasets import DATASET_DIR, TARGET_COL_SUFFIX

    merged_path = merged_path or (PROCESSED_DATA_DIR / "merged.parquet")
    feature_cols = _price_feature_columns(feature_version)
    df = pd.read_parquet(merged_path)
    logger.info(f"Rebuilding price_{feature_version} from {merged_path}: {df.shape}")
    df = build_forecast_columns(df)
    full_features = engineer_features(df, PRICE_FEATURES_MAX, validate=False)
    missing = [c for c in feature_cols if c not in full_features.columns]
    if missing:
        raise KeyError(
            f"{feature_version}: {len(missing)} frozen feature column(s) were not "
            f"recomputed from fresh merged data: {missing[:10]}"
        )

    target_col = f"target_price{TARGET_COL_SUFFIX}"
    dataset = full_features[feature_cols].copy()
    dataset[target_col] = df["target_price"].reindex(dataset.index)
    before = len(dataset)
    dataset = dataset.dropna()
    dropped = before - len(dataset)
    if dataset.empty:
        raise ValueError(f"Fresh price_{feature_version} dataset is empty after dropna")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASET_DIR / f"price_{feature_version}.parquet"
    dataset.to_parquet(out_path)
    logger.info(
        f"Fresh price_{feature_version}: {dataset.shape} -> {out_path} "
        f"(dropped {dropped}, last={dataset.index[-1]})"
    )
    return out_path


def _retrain_one_price_model(
    entry: dict,
    *,
    dataset_cache: dict[str, Path] | None = None,
    selection_step: str = "steady_retrain",
) -> str | None:
    """Retrain one price base model using stored hyperparams. Returns new run_id."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from energy_forecasting.modeling.cv import TimeSeriesSplitter
    from energy_forecasting.modeling.training import train_model
    from energy_forecasting.modeling.tuning import _make_model as _make_price_model

    feature_version = entry["feature_version"]
    config = entry["config"]
    dataset_cache = dataset_cache if dataset_cache is not None else {}
    if feature_version not in dataset_cache:
        dataset_cache[feature_version] = rebuild_price_dataset_from_merged(feature_version)
    ds_path = dataset_cache[feature_version]

    model = _make_price_model(config["model_type"], config["model_params"])
    tags = {
        "stage": "model_training",
        "feature_version": feature_version,
        "feature_contract": FEATURE_CONTRACT,
        "selection_step": selection_step,
        "dataset_rebuild": "fresh_merged",
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
    mode: str = "reweight",
) -> dict:
    """Retrain price models and rebuild the production ensemble config.

    ``mode="reweight"`` is the steady-state EP path: retrain the current
    positive-weight production members and recompute inverse-MAE weights on the
    recent holdout. ``mode="reselection"`` is the bootstrap/full-universe path:
    retrain every configured candidate, then re-run the category floor.
    """
    if mode not in {"reweight", "reselection"}:
        raise ValueError("mode must be 'reweight' or 'reselection'")

    if holdout_days is not None and holdout_days != HOLDOUT_DAYS:
        logger.warning(
            "holdout_days override is retained for CLI compatibility but retrain "
            "uses the holdout stored in each prediction artifact."
        )

    old_config = load_ensemble_config()
    old_mae = old_config.get("metrics", {}).get("mae", float("inf"))
    prod_names = set(production_model_names(old_config))
    retrain_names = (
        {entry["name"] for entry in old_config["models"]}
        if mode == "reselection"
        else prod_names
    )

    new_run_ids: dict[str, str] = {}
    dataset_cache: dict[str, Path] = {}
    selection_step = "bootstrap_reselection" if mode == "reselection" else "steady_retrain"
    for entry in old_config["models"]:
        if entry["name"] not in retrain_names:
            continue
        try:
            run_id = _retrain_one_price_model(
                entry,
                dataset_cache=dataset_cache,
                selection_step=selection_step,
            )
            if run_id:
                new_run_ids[entry["name"]] = run_id
        except Exception:
            logger.exception(f"Failed retrain for {entry['name']}")

    if not new_run_ids:
        raise RuntimeError("All price model retrains failed")

    new_config, new_mae = _build_retrain_ensemble(old_config, new_run_ids, mode=mode)

    if old_mae > 0:
        ratio = new_mae / old_mae
        needs_reselection = ratio > (1.0 + ENSEMBLE_DEGRADATION_THRESHOLD)
    else:
        ratio = 1.0
        needs_reselection = False

    _history_entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "mode": mode,
        "old_holdout_mae": round(old_mae, 3),
        "new_holdout_mae": round(new_mae, 3),
        "degradation_pct": round((new_mae / old_mae - 1) * 100, 1) if old_mae > 0 else 0.0,
        "n_models": len(retrain_names),
        "needs_reselection": needs_reselection,
    }

    if needs_reselection and not force:
        logger.warning(
            f"Retrain degraded: new MAE={new_mae:.3f} vs old={old_mae:.3f} "
            f"(ratio={ratio:.2%} > threshold {1 + ENSEMBLE_DEGRADATION_THRESHOLD:.0%}). "
            "Not updating config. Pass --force to override."
        )
        _append_retrain_history(_history_entry)
        return {
            "new_mae": new_mae,
            "old_mae": old_mae,
            "needs_reselection": True,
            "mode": mode,
            "config_path": None,
        }

    new_config["needs_reselection"] = needs_reselection
    new_config["retrain_mode"] = mode
    # default=str mirrors price.py's config write: a safety net so a stray
    # numpy scalar in the comparison/alignment blocks can never crash the ship
    # write after an expensive retrain (metrics are already coerced upstream).
    ENSEMBLE_CONFIG_PATH.write_text(json.dumps(new_config, indent=2, default=str))
    logger.info(f"ensemble_config.json updated: MAE {old_mae:.3f} -> {new_mae:.3f}")

    export_price_models(new_config)
    _append_retrain_history(_history_entry)

    return {
        "new_mae": new_mae,
        "old_mae": old_mae,
        "needs_reselection": needs_reselection,
        "mode": mode,
        "config_path": str(ENSEMBLE_CONFIG_PATH),
    }


def _build_retrain_ensemble(
    old_config: dict,
    new_run_ids: dict[str, str],
    *,
    mode: str,
) -> tuple[dict, float]:
    """Build a retrained ensemble config using the shared production builder."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from energy_forecasting.modeling.ensemble import (
        build_production_ensemble,
        ensemble_config_dict,
    )
    from energy_forecasting.modeling.price import _fetch_predictions, _ModelRun

    prod_names = set(production_model_names(old_config))
    if mode == "reselection":
        entries_for_build = list(old_config["models"])
        fixed_member_names = None
    else:
        entries_for_build = [entry for entry in old_config["models"] if entry["name"] in prod_names]
        fixed_member_names = {entry["name"] for entry in entries_for_build}

    model_runs = []
    for entry in entries_for_build:
        name = entry["name"]
        run_id = new_run_ids.get(name, entry["run_id"])
        model_runs.append(
            _ModelRun(
                name=name,
                run_id=run_id,
                model_type=entry["model_type"],
                feature_version=entry["feature_version"],
                config=entry["config"],
            )
        )

    production = build_production_ensemble(
        model_runs,
        prediction_loader=_fetch_predictions,
        strict_categories=(mode == "reselection"),
        candidate_model_names=fixed_member_names,
    )
    build_base_runs = {
        mr.name: {
            "run_id": mr.run_id,
            "model_type": mr.model_type,
            "feature_version": mr.feature_version,
            "config": mr.config,
        }
        for mr in model_runs
    }
    new_config = ensemble_config_dict(
        production.ensemble,
        base_runs=build_base_runs,
        metrics=production.metrics,
    )

    # Preserve the full candidate universe as configs. Only candidates present
    # in ``new_run_ids`` are fresh for this retrain mode.
    full_models = []
    for entry in old_config["models"]:
        updated = dict(entry)
        if entry["name"] in new_run_ids:
            updated["run_id"] = new_run_ids[entry["name"]]
        full_models.append(updated)
    new_config["models"] = full_models
    all_candidate_names = {entry["name"] for entry in old_config["models"]}
    all_candidates_fresh = mode == "reselection" and all_candidate_names <= set(new_run_ids)
    new_config["artifact_generation"] = {
        "mode": "bootstrap_reselection" if mode == "reselection" else "steady_state",
        "fresh_run_names": sorted(new_run_ids),
        "all_candidates_fresh": all_candidates_fresh,
    }
    new_config["ensemble_production"] = {
        "mode": mode,
        "selected_models": production.selected_models,
        "alignment": production.alignment_metadata,
        "candidate_metrics": production.candidate_metrics.to_dict(orient="records"),
    }
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
    mode: str = "reweight",
) -> dict:
    """Top-level retrain entry point.

    price_only=True is the default for GitHub Actions (gen/load retrain is manual).
    """
    result = run_price_retrain(force=force, holdout_days=holdout_days, mode=mode)

    if not price_only:
        logger.info(
            "Gen/load retrain takes 8-12 hours and must be run detached on the tower. "
            "Skipping in this run. Use 'make retrain-gen-load' for a detached run."
        )

    return result
