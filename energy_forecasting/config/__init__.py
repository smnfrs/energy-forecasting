"""Path constants and MLflow setup."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = PROCESSED_DATA_DIR / "cache"
LOCATIONS_DIR = Path(__file__).resolve().parent
HISTORICAL_FORECASTS_DIR = PROCESSED_DATA_DIR / "historical_forecasts"

# Raw subdirectories (created by stage 2)
SMARD_DIR = RAW_DATA_DIR / "smard"
WEATHER_DIR = RAW_DATA_DIR / "weather"
COMMODITIES_DIR = RAW_DATA_DIR / "commodities"
ENERGY_CHARTS_DIR = RAW_DATA_DIR / "energy_charts"

# Models
MODELS_DIR = PROJ_ROOT / "models"
MLFLOW_DB_PATH = PROJ_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# Deployment
DEPLOY_DIR = PROJ_ROOT / "deploy"
DEPLOY_DATA_DIR = DEPLOY_DIR / "data"
