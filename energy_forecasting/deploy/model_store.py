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
from loguru import logger

from energy_forecasting.config import MLFLOW_TRACKING_URI, MODELS_DIR

GEN_LOAD_MODELS_DIR = MODELS_DIR / "gen_load"
PRICE_MODELS_DIR = MODELS_DIR / "price"
GEN_LOAD_CONFIG_PATH = MODELS_DIR / "gen_load_config.json"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"

_WEIGHT_THRESHOLD = 1e-10  # SLSQP weights below this are treated as zero


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
        raise FileNotFoundError(
            f"{ENSEMBLE_CONFIG_PATH} not found. Run stage 5c training first."
        )
    with open(ENSEMBLE_CONFIG_PATH) as f:
        return json.load(f)


def _download_model(run_id: str) -> object:
    """Download and load the sklearn model artifact for one MLflow run."""
    client = _mlflow_client()
    local_dir = client.download_artifacts(run_id, "model")
    return mlflow.sklearn.load_model(local_dir)


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
        raise FileNotFoundError(
            f"{path} not found. Run 'energy-forecasting export-models' first."
        )
    return joblib.load(path)


# ── Price model export ────────────────────────────────────────────


def production_model_names(config: dict | None = None) -> list[str]:
    """Return names of ensemble members with non-zero weight."""
    if config is None:
        config = load_ensemble_config()
    weights = config["ensemble"]["weights"]
    return [name for name, w in weights.items() if abs(w) > _WEIGHT_THRESHOLD]


def export_price_models(config: dict | None = None) -> list[Path]:
    """Export production price models from MLflow to models/price/.

    Only exports non-zero-weight ensemble members.
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
            logger.info(f"  → {out_path}")
            written.append(out_path)
        except Exception:
            logger.exception(f"Failed to export price model {entry['name']}")

    return written


def load_price_model(run_id: str) -> object:
    """Load a price model from disk."""
    path = PRICE_MODELS_DIR / f"{run_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'energy-forecasting export-models' first."
        )
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
