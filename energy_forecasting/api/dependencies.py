"""FastAPI dependency layer — loads pre-computed JSON from deploy/data/.

The API is deliberately stateless: it serves static files produced by the
daily inference pipeline. No model loading happens at request time.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from energy_forecasting.config import DEPLOY_DATA_DIR

GEN_LOAD_DATA_DIR = DEPLOY_DATA_DIR / "gen_load"
PRICE_FORECAST_PATH = DEPLOY_DATA_DIR / "price_forecast.json"
HISTORY_PATH = DEPLOY_DATA_DIR / "forecast_history.json"
METADATA_PATH = DEPLOY_DATA_DIR / "model_metadata.json"
ERRORS_DIR = DEPLOY_DATA_DIR / "errors"

_GEN_LOAD_FILES = {
    "wind_onshore": GEN_LOAD_DATA_DIR / "wind_onshore_national.json",
    "wind_offshore": GEN_LOAD_DATA_DIR / "wind_offshore_national.json",
    "solar": GEN_LOAD_DATA_DIR / "solar_national.json",
    "load": GEN_LOAD_DATA_DIR / "load_national.json",
}


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return json.loads(path.read_text())


def get_price_forecast() -> dict:
    return _load_json(PRICE_FORECAST_PATH)


def get_gen_load_forecast(target: str) -> dict:
    if target not in _GEN_LOAD_FILES:
        raise ValueError(
            f"Unknown gen/load target '{target}'. "
            f"Available: {list(_GEN_LOAD_FILES)}"
        )
    return _load_json(_GEN_LOAD_FILES[target])


def get_forecast_history() -> dict:
    return _load_json(HISTORY_PATH)


def get_model_metadata() -> dict:
    return _load_json(METADATA_PATH)


def get_daily_errors(days: int | None = None) -> list[dict]:
    if not ERRORS_DIR.exists():
        return []
    files = sorted(ERRORS_DIR.glob("*.json"), reverse=True)
    if days is not None:
        files = files[:days]
    errors = []
    for f in files:
        try:
            errors.append(json.loads(f.read_text()))
        except Exception:
            pass
    return errors


def is_data_available() -> bool:
    return PRICE_FORECAST_PATH.exists()


def count_model_files() -> int:
    from energy_forecasting.deploy.model_store import PRICE_MODELS_DIR

    if not PRICE_MODELS_DIR.exists():
        return 0
    return len(list(PRICE_MODELS_DIR.glob("*.joblib")))
