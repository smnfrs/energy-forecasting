"""Model serialization: MLflow → disk and disk → inference.

Handles two model families:
- Gen/load: one joblib per (target, region), referenced by gen_load_config.json
- Price: one joblib per ensemble base model run_id, referenced by ensemble_config.json

All artifacts are stored under models/gen_load/ and models/price/ respectively.
These directories are gitignored; only the JSON config files are version-controlled.
Models are uploaded to a GitHub Release and downloaded by CI at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from loguru import logger

from energy_forecasting.config import MLFLOW_TRACKING_URI, MODELS_DIR

GEN_LOAD_MODELS_DIR = MODELS_DIR / "gen_load"
PRICE_MODELS_DIR = MODELS_DIR / "price"
GEN_LOAD_CONFIG_PATH = MODELS_DIR / "gen_load_config.json"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"
PRICE_FEATURE_COLS_PATH = MODELS_DIR / "price_feature_cols.json"

_WEIGHT_THRESHOLD = 1e-10  # Tiny ensemble weights below this are treated as zero


def _mlflow_client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.MlflowClient()


def load_gen_load_config() -> dict:
    if not GEN_LOAD_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{GEN_LOAD_CONFIG_PATH} not found. "
            "Run 'energy-forecasting train gen-load' or "
            "'energy-forecasting gen-load-config' first."
        )
    with open(GEN_LOAD_CONFIG_PATH) as f:
        return json.load(f)


def load_ensemble_config() -> dict:
    if not ENSEMBLE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"{ENSEMBLE_CONFIG_PATH} not found. Run stage 5c training first.")
    with open(ENSEMBLE_CONFIG_PATH) as f:
        return json.load(f)


def _download_model(run_id: str) -> object:
    """Download and load the sklearn model artifact for one MLflow run."""
    client = _mlflow_client()
    local_dir = client.download_artifacts(run_id, "model")
    return mlflow.sklearn.load_model(local_dir)


def _download_scaler(run_id: str) -> object | None:
    """Download the separate scaler artifact for one MLflow run, if it exists.

    Some models (Ridge/Lasso with sample_weight) use the pre-scale workaround:
    X is scaled externally before MAPIE, so the scaler is saved as a separate
    MLflow artifact named "scaler". Returns None if no scaler artifact exists.
    """
    client = _mlflow_client()
    try:
        local_dir = client.download_artifacts(run_id, "scaler")
        return mlflow.sklearn.load_model(local_dir)
    except Exception:
        return None


# ── Gen/load export ───────────────────────────────────────────────


def export_gen_load_models(config: dict | None = None) -> list[Path]:
    """Export all gen/load models from MLflow to models/gen_load/.

    Parameters
    ----------
    config : gen_load_config dict. Loaded from disk if not provided.

    Returns list of written joblib paths.
    """
    if config is None:
        config = load_gen_load_config()

    GEN_LOAD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for combo_key, entry in config["combos"].items():
        target, region = combo_key.split("/")
        run_id = entry["run_id"]
        out_path = GEN_LOAD_MODELS_DIR / f"{target}_{region}.joblib"

        logger.info(f"Exporting gen/load model {combo_key} (run {run_id[:8]})")
        try:
            model = _download_model(run_id)
            joblib.dump(model, out_path)
            logger.info(f"  → {out_path}")
            written.append(out_path)
        except Exception:
            logger.exception(f"Failed to export gen/load model for {combo_key}")

    return written


def load_gen_load_model(target: str, region: str) -> object:
    """Load a gen/load model from disk."""
    path = GEN_LOAD_MODELS_DIR / f"{target}_{region}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'energy-forecasting export-models' first.")
    return joblib.load(path)


# ── Price model export ────────────────────────────────────────────


def production_model_names(config: dict | None = None) -> list[str]:
    """Return names of ensemble members with non-zero weight."""
    if config is None:
        config = load_ensemble_config()
    weights = config["ensemble"]["weights"]
    return [name for name, w in weights.items() if abs(w) > _WEIGHT_THRESHOLD]


def _dataset_feature_columns(feature_version: str) -> list[str]:
    """Read and validate persisted feature columns for a price feature version."""
    from energy_forecasting.features.validation import validate_price_feature_list
    from energy_forecasting.modeling.datasets import DATASET_DIR

    ds_path = DATASET_DIR / f"price_{feature_version}.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"{ds_path} not found. Regenerate price datasets before exporting price models."
        )

    try:
        import pyarrow.parquet as pq

        names = pq.read_schema(ds_path).names
    except Exception:
        names = list(pd.read_parquet(ds_path).columns)

    feature_cols = [c for c in names if not c.endswith("__target") and c != "__index_level_0__"]
    validate_price_feature_list(feature_cols)
    return feature_cols


def export_price_feature_columns(config: dict | None = None) -> Path:
    """Write models/price_feature_cols.json for production price feature versions."""
    if config is None:
        config = load_ensemble_config()

    prod_names = set(production_model_names(config))
    feature_versions = sorted(
        {entry["feature_version"] for entry in config["models"] if entry["name"] in prod_names}
    )
    payload = {fv: _dataset_feature_columns(fv) for fv in feature_versions}

    PRICE_FEATURE_COLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRICE_FEATURE_COLS_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote price feature columns for {feature_versions} -> {PRICE_FEATURE_COLS_PATH}")
    return PRICE_FEATURE_COLS_PATH


def export_price_models(config: dict | None = None) -> list[Path]:
    """Export production price models from MLflow to models/price/.

    Only exports non-zero-weight ensemble members.
    For models that use the pre-scale workaround (Ridge/Lasso), also exports
    the separate scaler artifact as {run_id}_scaler.joblib.
    """
    if config is None:
        config = load_ensemble_config()

    PRICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prod_names = set(production_model_names(config))

    written: list[Path] = []
    for entry in config["models"]:
        if entry["name"] not in prod_names:
            continue
        run_id = entry["run_id"]
        out_path = PRICE_MODELS_DIR / f"{run_id}.joblib"

        logger.info(f"Exporting price model {entry['name']} (run {run_id[:8]})")
        try:
            model = _download_model(run_id)
            joblib.dump(model, out_path)
            logger.info(f"  -> {out_path}")
            written.append(out_path)

            scaler = _download_scaler(run_id)
            if scaler is not None:
                scaler_path = PRICE_MODELS_DIR / f"{run_id}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
                logger.info(f"  -> {scaler_path} (pre-scale workaround)")
                written.append(scaler_path)
        except Exception:
            logger.exception(f"Failed to export price model {entry['name']}")

    for existing in PRICE_MODELS_DIR.glob("*.joblib"):
        if existing not in written:
            existing.unlink()
            logger.info(f"Removed stale price model artifact {existing}")

    feature_cols_path = export_price_feature_columns(config)
    written.append(feature_cols_path)
    validate_price_artifact_lockstep(config)
    return written


def validate_price_artifact_lockstep(config: dict | None = None) -> dict[str, list[str]]:
    """Validate exported price artifacts match the production ensemble config.

    This guards against fresh configs being paired with stale model files or
    stale feature-column manifests. Scaler artifacts are optional because only
    runs that used the pre-scale workaround export them.
    """
    if config is None:
        config = load_ensemble_config()

    prod_names = set(production_model_names(config))
    prod_entries = [entry for entry in config["models"] if entry["name"] in prod_names]
    missing_models = [
        str(PRICE_MODELS_DIR / f"{entry['run_id']}.joblib")
        for entry in prod_entries
        if not (PRICE_MODELS_DIR / f"{entry['run_id']}.joblib").exists()
    ]

    missing_feature_versions: list[str] = []
    if PRICE_FEATURE_COLS_PATH.exists():
        feature_cols = json.loads(PRICE_FEATURE_COLS_PATH.read_text())
    else:
        feature_cols = {}
    for fv in sorted({entry["feature_version"] for entry in prod_entries}):
        if fv not in feature_cols:
            missing_feature_versions.append(fv)

    report = {
        "missing_models": missing_models,
        "missing_feature_versions": missing_feature_versions,
    }
    if missing_models or missing_feature_versions:
        raise RuntimeError(f"Price artifact lockstep validation failed: {report}")
    logger.info(
        f"Price artifact lockstep OK: {len(prod_entries)} production models, "
        f"{len(set(e['feature_version'] for e in prod_entries))} feature version(s)"
    )
    return report


def load_price_model_scaler(run_id: str) -> object | None:
    """Load the separate scaler for a price model, or None if it has none."""
    path = PRICE_MODELS_DIR / f"{run_id}_scaler.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_price_model(run_id: str) -> object:
    """Load a price model from disk."""
    path = PRICE_MODELS_DIR / f"{run_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'energy-forecasting export-models' first.")
    return joblib.load(path)


def export_all_models() -> None:
    """Export all production models (gen/load + price) from MLflow to disk."""
    logger.info("Exporting gen/load models...")
    gl_written = export_gen_load_models()
    logger.info(f"Gen/load: {len(gl_written)} models exported")

    logger.info("Exporting price models...")
    pr_written = export_price_models()
    logger.info(f"Price: {len(pr_written)} models exported")

    logger.info(f"Total exported: {len(gl_written) + len(pr_written)} models")
