## Stage 1: Foundation

**Goal:** Project skeleton with working environment, config patterns established, utility functions ready. Everything in this stage is infrastructure that all subsequent stages build on.

**Source material:**
- EP: `src/config/`, `src/api/schemas.py`, `src/data/processing.py` config dicts, `pyproject.toml`, `Makefile`
- EMA: `data_collection_modules/parquet_operations.py`, `eu_locations.py`, `Pipfile`

---

### 1.1 Project Structure

```
energy-forecasting/
├── energy_forecasting/
│   ├── __init__.py
│   ├── cli.py                    # Typer CLI entry point
│   ├── config/
│   │   ├── __init__.py           # Path constants, MLflow setup
│   │   ├── columns.py            # Short name registry + SMARD filter key mappings
│   │   ├── cleaning.py           # Cleaning rule definitions (Python dataclasses)
│   │   ├── availability.py       # Leakage/availability rules
│   │   ├── features.py           # Feature list definitions per target
│   │   ├── modeling.py           # Blend constants, experiment names, category matchers
│   │   └── smard.py              # SMARD API key/region/TSO mappings
│   ├── data/
│   │   ├── sources.py            # DataSource base class + all source implementations
│   │   ├── smard.py              # SMARD API client functions
│   │   ├── commodities.py        # Commodity download, gap reconstruction, bias correction
│   │   ├── weather.py            # Open-Meteo client (three-endpoint logic)
│   │   ├── processing.py         # Combine, merge, missing value handling
│   │   └── io.py                 # save_parquet, load_parquet (compression, dtype reduction)
│   ├── features/
│   │   ├── parser.py             # Suffix DSL parser
│   │   ├── engine.py             # Feature computation dispatcher
│   │   ├── market.py             # Market feature functions (spreads, EWMA, rolling, lags, temporal)
│   │   ├── weather_wind.py       # WeatherWindPowerFE class
│   │   ├── weather_solar.py      # WeatherSolarPowerFE class
│   │   ├── weather_load.py       # WeatherLoadFE class
│   │   ├── spatial.py            # Spatial aggregation (haversine, IDW, capacity-weighted)
│   │   ├── validation.py         # Leakage validation (checks suffix against availability)
│   │   └── cache.py              # Hash-based dataset caching
│   ├── modeling/
│   │   ├── training.py           # train_and_log, time-series CV, sample weighting
│   │   ├── tuning.py             # Optuna integration (GridSampler + TPE), search spaces
│   │   ├── blend.py              # Inverse-MAE blend (select, validate, train, weight)
│   │   ├── stacking.py           # Stacking meta-learner ensemble
│   │   ├── ensemble.py           # Auto-selection wrapper (blend vs stacking on holdout)
│   │   ├── intervals.py          # MAPIE wrapping, PI blending, coverage tracking
│   │   ├── metrics.py            # Metric calculations (RMSE, MAE, skill scores, PI coverage)
│   │   └── baselines.py          # Naive, ARIMA, ETS baselines
│   ├── deploy/
│   │   ├── inference.py          # Daily inference pipeline
│   │   ├── retrain.py            # Periodic retrain pipeline
│   │   └── publish.py            # Forecast -> JSON/CSV for API/dashboard
│   ├── api/
│   │   ├── app.py                # FastAPI app
│   │   ├── routes.py             # API endpoints
│   │   ├── schemas.py            # Pydantic response models (defined early, stage 1)
│   │   └── dependencies.py       # Data loading helpers
│   └── mlflow_utils.py           # TrackedRun wrapper, audit, archive, compare helpers
├── data/                         # (gitignored)
│   ├── raw/
│   │   ├── smard/{region}/       # e.g. smard/DE_LU/, smard/DE_50HZ/
│   │   ├── weather/{asset_type}/{tso}/  # e.g. weather/offshore/50Hertz/
│   │   ├── commodities/
│   │   └── energy_charts/
│   ├── locations/
│   │   └── eu_locations.json     # Extracted from EMA's eu_locations.py
│   └── processed/
│       ├── merged.parquet
│       └── cache/                # Hash-based feature dataset cache
├── models/                       # (gitignored — production models in GitHub Releases)
│   └── mlflow.db                 # Local MLflow tracking database
├── deploy/
│   ├── index.html                # Dashboard
│   ├── script.js
│   ├── assets/
│   └── data/                     # Generated forecast JSON/CSV
├── tests/
│   ├── test_io.py
│   ├── test_columns.py
│   ├── test_cleaning_rules.py
│   ├── test_availability.py
│   ├── test_schemas.py
│   └── conftest.py
├── .github/workflows/
│   ├── daily_forecast.yml
│   └── retrain.yml
├── docs/
│   ├── master_plan.md
│   └── archive/
├── Makefile
├── pyproject.toml
├── .gitignore
└── README.md
```

---

### 1.2 Environment & Dependencies

**`pyproject.toml`** — consolidated from EP (`flit_core` + `pyproject.toml`) and EMA (`Pipfile`).

```toml
[build-system]
requires = ["flit_core>=3.2,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "energy-forecasting"
version = "0.1.0"
description = "Day-ahead electricity price and generation/load forecasting for the German energy market"
requires-python = ">=3.13"

dependencies = [
    # Data
    "pandas>=2.2",
    "pyarrow>=15.0",
    "numpy>=2.0",

    # ML models
    "scikit-learn>=1.5",
    "lightgbm>=4.0",
    "xgboost>=3.0",
    "catboost>=1.2",
    "mapie>=1.3",

    # Experiment tracking & tuning
    "mlflow>=3.0",
    "optuna>=4.0",

    # API & CLI
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "typer>=0.15",
    "httpx>=0.27",

    # Data sources
    "requests>=2.31",
    "fredapi>=0.5",
    "yfinance>=0.2",

    # Domain
    "pysolar>=0.10",
    "holidays>=0.40",

    # Utilities
    "loguru>=0.7",
    "joblib>=1.4",
    "tqdm>=4.66",
    "python-dotenv>=1.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.9",
    "pytest>=8.0",
    "jupyterlab",
    "matplotlib>=3.9",
]

[tool.ruff]
line-length = 99
src = ["energy_forecasting"]

[tool.ruff.lint]
extend-select = ["I"]  # isort

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Key dependency notes:**
- `mapie>=1.3` — uses `CrossConformalRegressor` API (0.9.x `MapieRegressor` was removed in 1.0)
- `prophet` and `statsmodels` removed from core deps (only needed for baselines, add to optional later)
- EMA's `openmeteo-requests`, `requests-cache`, `retry-requests` deferred to stage 2 (only needed for weather collection)
- All deps tested compatible with Python 3.13

---

### 1.3 Config: Path Constants & MLflow Setup

**`energy_forecasting/config/__init__.py`** — ported from EP's `src/config/__init__.py`.

```python
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = PROCESSED_DATA_DIR / "cache"
LOCATIONS_DIR = DATA_DIR / "locations"

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
```

Changes from EP:
- Removed `INTERIM_DATA_DIR` (no CSV intermediate stage)
- Removed `EMA_DATA_DIR` / `EMA_HISTORICAL_FORECASTS_DIR` (the cross-repo bridge is eliminated)
- Added `WEATHER_DIR`, `CACHE_DIR`, `LOCATIONS_DIR`
- Removed `get_path()` dispatcher — direct path references are clearer

---

### 1.4 Config: Short Name Registry & SMARD Mappings

**`energy_forecasting/config/columns.py`** — the DSL's lookup table. Maps concise names (used in feature strings like `price_d1`) to actual DataFrame column names.

Ported from EP's `src/config/smard.py` (`camel_dict`, `filter_dict`, `clean_filename`). The SMARD filter key → German description → snake_case column name pipeline is preserved, but the short name registry is a new layer on top.

```python
# ── Short name registry ──────────────────────────────────────────────
# Used by the suffix DSL parser. Keys are the concise names that appear
# in feature strings. Values are the actual DataFrame column names
# (snake_case German, matching SMARD's naming after clean_filename).

SHORT_NAMES: dict[str, str] = {
    # Target
    "price":            "target_price",

    # Generation (actuals)
    "gen_wind_on":      "stromerzeugung_wind_onshore",
    "gen_wind_off":     "stromerzeugung_wind_offshore",
    "gen_solar":        "stromerzeugung_photovoltaik",
    "gen_nuclear":      "stromerzeugung_kernenergie",
    "gen_lignite":      "stromerzeugung_braunkohle",
    "gen_coal":         "stromerzeugung_steinkohle",
    "gen_gas":          "stromerzeugung_erdgas",
    "gen_hydro":        "stromerzeugung_wasserkraft",
    "gen_biomass":      "stromerzeugung_biomasse",
    "gen_pumped":       "stromerzeugung_pumpspeicher",
    "gen_other":        "stromerzeugung_sonstige_konventionelle",
    "gen_other_renew":  "stromerzeugung_sonstige_erneuerbare",

    # Forecasts (published for today — no lag required)
    "prog_load":        "prognostizierter_verbrauch_gesamt",
    "prog_gen_total":   "prognostizierte_erzeugung_gesamt",
    "prog_gen_wind_pv": "prognostizierte_erzeugung_wind_und_photovoltaik",
    "prog_gen_wind_on": "prognostizierte_erzeugung_onshore",
    "prog_gen_wind_off":"prognostizierte_erzeugung_offshore",
    "prog_gen_solar":   "prognostizierte_erzeugung_photovoltaik",
    "prog_gen_other":   "prognostizierte_erzeugung_sonstige",
    "prog_residual":    "prognostizierte_residuallast",

    # Consumption
    "load":             "stromverbrauch_gesamt_(netzlast)",
    "residual_load":    "residuallast",

    # Commodities
    "carbon":           "carbon_eur_per_ton",
    "carbon_rt":        "carbon_realtime_eur_per_ton",
    "ttf":              "ttf_eur_per_mwh",
    "brent":            "brent_usd_per_barrel",

    # Neighbour prices (for spread computation)
    "price_fr":         "marktpreis_frankreich",
    "price_nl":         "marktpreis_niederlande",
    "price_at":         "marktpreis_oesterreich",
    "price_dk1":        "marktpreis_daenemark_1",
    "price_dk2":        "marktpreis_daenemark_2",
    "price_cz":         "marktpreis_tschechien",
    "price_pl":         "marktpreis_polen",
    "price_ch":         "marktpreis_schweiz",
    "price_no2":        "marktpreis_norwegen_2",
    "price_se4":        "marktpreis_schweden_4",
    "price_be":         "marktpreis_belgien",
    "price_hu":         "marktpreis_ungarn",
    "price_si":         "marktpreis_slowenien",
    "price_it_n":       "marktpreis_italien_nord",

    # Derived (computed during feature engineering, not raw columns)
    # These are registered so the DSL knows they exist.
    # Actual computation is in features/market.py
    "net_export_fr":    "_derived_net_export_frankreich",
    "net_export_nl":    "_derived_net_export_niederlande",
    "net_export_at":    "_derived_net_export_oesterreich",
    # ... (one per flow pair from EP's FLOW_PAIRS)
    "spread_fr":        "_derived_spread_frankreich",
    "spread_nl":        "_derived_spread_niederlande",
    # ... (one per neighbour)
    "gen_pct_wind":     "_derived_gen_pct_wind",
    "gen_pct_solar":    "_derived_gen_pct_solar",
    # ... (one per source)

    # Weather-derived (computed by weather FE classes, stage 4)
    # Registered here so the DSL can reference them.
    # Prefixed with asset type and aggregation method.
    "wpd_offshore_cap": "_derived_wpd_offshore_cap_weighted",
    "wpd_onshore_cap":  "_derived_wpd_onshore_cap_weighted",
    "temp_cities_pop":  "_derived_temperature_cities_pop_weighted",
    "ghi_solar_cap":    "_derived_ghi_solar_cap_weighted",
}

REVERSE_SHORT_NAMES: dict[str, str] = {v: k for k, v in SHORT_NAMES.items()}


# ── SMARD filter key mappings ────────────────────────────────────────
# Ported from EP's src/config/smard.py. Maps SMARD API integer filter
# keys to German descriptions and snake_case column names.

def clean_column_name(description: str) -> str:
    """Convert German SMARD description to snake_case column name.

    'Stromerzeugung: Braunkohle' -> 'stromerzeugung_braunkohle'
    'Cross-border Flows: France (Exports)' -> 'cross-border_flows_france_exports'
    """
    # Port EP's clean_filename() logic
    ...

# Full SMARD filter_dict (96 entries) ported from EP's src/config/smard.py.
# Not reproduced here for brevity — copy wholesale from EP.
SMARD_FILTER_KEYS: dict[int, str] = {
    1223: "Stromerzeugung: Braunkohle",
    1224: "Stromerzeugung: Kernenergie",
    1225: "Stromerzeugung: Wind Offshore",
    # ... (full 96-entry dict from EP)
}

SMARD_COLUMN_NAMES: dict[int, str] = {
    k: clean_column_name(v) for k, v in SMARD_FILTER_KEYS.items()
}

# Cross-border flow registries (from EP's smard.py)
CROSS_BORDER_DE_LU: dict[int, str] = { ... }      # 23 entries
CROSS_BORDER_DE_AT_LU: dict[int, str] = { ... }    # 24 entries

# Keys to exclude from data downloads
EXCLUDED_KEYS: set[int] = { ... }  # Installed capacity + scheduled commercial
```

**What's new vs EP:** The `SHORT_NAMES` registry is a new layer. EP had `camel_dict` (SMARD key → column name) but no concise aliases. The short names serve the DSL — `price_d1` is far more readable than `target_price_d1`, and `gen_wind_on_d7_d1` beats `stromerzeugung_wind_onshore_d7_d1`.

---

### 1.5 Config: Cleaning Rules

**`energy_forecasting/config/cleaning.py`** — the cleaning pipeline as a function that calls small helper functions. Each step is a function call with a comment explaining the domain rationale. The helper functions live in `energy_forecasting/data/processing.py` (stage 3).

This is config (what to clean and why), not logic (how to clean). The helpers that implement each operation (`fill_zero_after`, `clip_bounds`, etc.) are ~10-15 lines each in `data/processing.py`.

```python
"""Cleaning pipeline configuration.

Processing order matters: drop → physical bounds → structural fills
→ calculated fills → correlate fills → interpolation.

Ported from EP's handle_missing_values() and EMA's physical validation.
"""

from energy_forecasting.data.processing import (
    clip_bounds,
    drop_columns,
    fill_from_column,
    fill_from_difference,
    fill_gen_total,
    fill_zero_after,
    fill_zero_before,
    fill_zero_before_first_valid,
    interpolate_gaps,
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules to the merged dataset.

    Each step has a comment documenting the domain rationale.
    Order matters — see module docstring.
    """

    # ── Drop redundant columns ──────────────────────────────────────
    # Values already captured in target_price
    df = drop_columns(df, [
        "marktpreis_deutschland_luxemburg",
        "marktpreis_deutschland_oesterreich_luxemburg",
        "marktpreis_anrainer_de_lu",
    ])

    # ── Physical bounds (weather + market) ──────────────────────────
    # Ported from EMA's phys_limits; extended for market data
    df = clip_bounds(df, "temperature_2m_*", min=-45, max=50)
    df = clip_bounds(df, "wind_speed_*", min=0, max=200)
    df = clip_bounds(df, "relative_humidity_*", min=0, max=100)
    df = clip_bounds(df, "shortwave_radiation_*", min=0, max=1400)
    df = clip_bounds(df, "direct_radiation_*", min=0, max=1400)
    df = clip_bounds(df, "cloud_cover_*", min=0, max=100)
    df = clip_bounds(df, "target_price", min=-500, max=1000, action="nan")
    df = clip_bounds(df, "stromverbrauch_*", min=0)

    # ── Structural zero fills ───────────────────────────────────────
    # Nuclear: decommissioned April 2023
    df = fill_zero_after(df, "stromerzeugung_kernenergie", after="last_valid")

    # Austria neighbour flows: irrelevant after DE-AT-LU split
    df = fill_zero_after(df, [
        "cross-border_flows_hungary_exports",
        "cross-border_flows_hungary_imports",
        "cross-border_flows_slovenia_exports",
        "cross-border_flows_slovenia_imports",
    ], after="2018-09-30T22:00:00Z")

    # Austria direct flows/price: didn't exist before split
    df = fill_zero_before(df, [
        "cross-border_flows_austria_exports",
        "cross-border_flows_austria_imports",
        "marktpreis_oesterreich",
    ], before="2018-09-30T22:00:00Z")

    # Belgium flows: reporting started Oct 2017
    df = fill_zero_before(df, [
        "cross-border_flows_belgium_exports",
        "cross-border_flows_belgium_imports",
    ], before="2017-10-10T22:00:00Z")

    # Norway flows: reporting started late
    df = fill_zero_before_first_valid(df, [
        "cross-border_flows_norway_2_exports",
        "cross-border_flows_norway_2_imports",
    ])

    # ── Calculated fills ────────────────────────────────────────────
    # Other forecast = total - wind+PV forecast
    df = fill_from_difference(df, "prognostizierte_erzeugung_sonstige",
                              total="prognostizierte_erzeugung_gesamt",
                              subtract="prognostizierte_erzeugung_wind_und_photovoltaik")

    # Load forecast backfilled from actual load (r=0.97)
    df = fill_from_column(df, "prognostizierter_verbrauch_gesamt",
                          source="stromverbrauch_gesamt_(netzlast)")

    # Generation total: complex logic (30-day recency check, component sum fallback)
    df = fill_gen_total(df)

    # Poland/Switzerland prices: zero-spread assumption
    df = fill_from_column(df, ["marktpreis_polen", "marktpreis_schweiz"],
                          source="target_price")

    # ── Final interpolation ─────────────────────────────────────────
    # Cubic spline for remaining small gaps (max 5 consecutive hours)
    df = interpolate_gaps(df, method="cubicspline", max_gap=5,
                          exclude=["regime_de_at_lu", "regime_quarter_hourly",
                                   "target_price"])

    return df
```

**What's new vs EP:** EP had these as procedural if-blocks in `handle_missing_values()` (~200 lines) with the logic and config interleaved. Here the config (`cleaning.py`) is separated from the logic (`processing.py`) — the config says *what* to clean and *why* (via comments), the helper functions say *how*. Adding a new rule is one function call with a comment.

---

### 1.6 Config: Availability Rules

**`energy_forecasting/config/availability.py`** — declares when each data source is physically available for prediction. Used by leakage validation (stage 4) to check that feature suffixes imply safe lags.

```python
from dataclasses import dataclass


@dataclass
class AvailabilityRule:
    """When is this column's data available for prediction?

    max_offset_days: how many days back is the latest available data?
        0 = today's data is available (forecasts, deterministic features)
       -1 = yesterday's data is the latest available
       -2 = two days ago (business day delay)
    cutoff_hour: hour (UTC) by which the data is published on the offset day.
        None = available all day (e.g., auction results published once).
    """
    pattern: str               # fnmatch pattern against short names
    max_offset_days: int
    cutoff_hour: int | None
    reason: str


AVAILABILITY_RULES: list[AvailabilityRule] = [
    # Forecasts — published for today, available before the auction
    AvailabilityRule("prog_*", 0, None,
                     "TSO forecasts published for delivery day"),
    AvailabilityRule("hour_*", 0, None,
                     "Deterministic temporal features"),
    AvailabilityRule("dow_*", 0, None,
                     "Deterministic temporal features"),
    AvailabilityRule("is_holiday", 0, None,
                     "Deterministic"),
    AvailabilityRule("day_index", 0, None,
                     "Deterministic"),

    # Weather forecasts — available for today (Open-Meteo forecast endpoint)
    AvailabilityRule("wpd_*", 0, None,
                     "Weather forecasts available for delivery day"),
    AvailabilityRule("temp_*", 0, None,
                     "Weather forecasts available for delivery day"),
    AvailabilityRule("ghi_*", 0, None,
                     "Weather forecasts available for delivery day"),

    # Price — D-1 auction results published previous afternoon
    AvailabilityRule("price", -1, None,
                     "EPEX SPOT auction results published D-1 ~13:00 CET"),
    AvailabilityRule("price_*", -1, None,
                     "Neighbour prices published D-1"),
    AvailabilityRule("spread_*", -1, None,
                     "Derived from prices, same availability"),

    # Generation/load actuals — SMARD publishes with ~10h delay
    AvailabilityRule("gen_*", -1, 10,
                     "SMARD generation actuals published by ~11:00 CET"),
    AvailabilityRule("load", -1, 10,
                     "SMARD load actuals same delay as generation"),
    AvailabilityRule("net_export_*", -1, 10,
                     "Derived from cross-border flows, same delay"),
    AvailabilityRule("gen_pct_*", -1, 10,
                     "Derived from generation, same delay"),

    # Commodities — business day delay
    AvailabilityRule("carbon", -2, None,
                     "ICAP carbon published with ~2 day lag"),
    AvailabilityRule("carbon_rt", -1, None,
                     "CO2.L equity proxy, previous close"),
    AvailabilityRule("ttf", -2, None,
                     "TTF futures, business day delay"),
    AvailabilityRule("brent", -2, None,
                     "Brent futures, business day delay"),
]
```

---

### 1.7 Config: Modelling Constants

**`energy_forecasting/config/modeling.py`** — experiment names, model categories, ensemble constants. Ported from EP's `src/config/modeling.py`.

```python
# MLflow experiment names (decision #2)
EXPERIMENTS = {
    "price_feature_selection": "price/feature_selection",
    "price_model_training":    "price/model_training",
    "price_production":        "price/production",
    "gen_wind_onshore":        "generation/wind_onshore",
    "gen_wind_offshore":       "generation/wind_offshore",
    "gen_solar":               "generation/solar",
    "gen_load":                "generation/load",
}

# Model categories for blend candidate selection
MODEL_CATEGORIES = {
    "linear":   ["Ridge", "Lasso", "ElasticNet"],
    "lgbm":     ["LGBMRegressor"],
    "xgboost":  ["XGBRegressor"],
    "catboost": ["CatBoostRegressor"],
}

# Blend defaults
BLEND_HOLDOUT_DAYS = 90
BLEND_CV_FOLDS = 5
BLEND_DEGRADATION_THRESHOLD = 0.20  # 20% MAE increase triggers flag

# Ensemble comparison: methods to evaluate at each retrain
ENSEMBLE_METHODS = ["inverse_mae_blend", "stacking"]
```

---

### 1.8 Parquet I/O Utility

**`energy_forecasting/data/io.py`** — ported from EMA's `ParquetOperations` class, simplified to two functions.

```python
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def reduce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 -> float32, int64 -> int32 where values fit.

    Returns a new DataFrame with reduced dtypes. Does not mutate the input.
    Ported from EMA's ParquetOperations._reduce_dtypes().
    """
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        if df[col].isna().all():
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
            df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype("int32")
    return df


def save_parquet(
    df: pd.DataFrame,
    path: Path | str,
    compress: bool = True,
    downcast: bool = True,
) -> None:
    """Save DataFrame to Parquet with optional zstd compression and dtype reduction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if downcast:
        df = reduce_dtypes(df)

    compression = "zstd" if compress else None
    df.to_parquet(path, engine="pyarrow", compression=compression)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.debug(f"Saved {path.name}: {len(df)} rows, {size_mb:.1f} MB")


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load Parquet file."""
    return pd.read_parquet(Path(path), engine="pyarrow")
```

---

### 1.9 API Data Contract (Pydantic Schemas)

**`energy_forecasting/api/schemas.py`** — defines the output shape that inference produces and the API serves. Ported from EP's `src/api/schemas.py`, extended for gen/load targets and prediction intervals.

```python
from datetime import datetime
from pydantic import BaseModel


# ── Forecast schemas ─────────────────────────────────────────────────

class HourlyForecast(BaseModel):
    """Single hourly forecast point with optional prediction interval."""
    timestamp: datetime
    forecast: float
    forecast_lower: float | None = None
    forecast_upper: float | None = None


class ForecastResponse(BaseModel):
    """Complete forecast for a single target and region."""
    target: str                    # "price", "wind_onshore", "solar", "load"
    region: str                    # "DE_LU", "DE_50HZ", "national", etc.
    issued_at: datetime
    horizon_hours: int             # 24 for price, 168 for gen/load
    unit: str                      # "EUR/MWh", "MW"
    forecasts: list[HourlyForecast]
    ensemble_method: str | None = None  # "blend" or "stacking"
    model_count: int | None = None


class ForecastHistoryResponse(BaseModel):
    """Historical forecasts for review/backtesting."""
    target: str
    forecasts: list[ForecastResponse]
    count: int


# ── Model/performance schemas ────────────────────────────────────────

class ModelInfo(BaseModel):
    """Individual model in an ensemble."""
    name: str
    category: str                  # "lgbm", "xgboost", "catboost", "linear"
    weight: float


class ModelsResponse(BaseModel):
    """Ensemble composition and metrics."""
    target: str
    ensemble_method: str
    models: list[ModelInfo]
    holdout_mae: float
    holdout_rmse: float
    pi_coverage: float | None = None
    last_retrain: datetime


class DailyError(BaseModel):
    """One day's forecast error."""
    date: datetime
    mae: float
    rmse: float


class PerformanceResponse(BaseModel):
    """Forecast accuracy over time."""
    target: str
    blend_errors: list[DailyError]
    pi_coverage_30d: float | None = None


# ── Health ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health check."""
    status: str                    # "healthy", "degraded", "unhealthy"
    models_loaded: int
    data_available: bool
    last_data_update: datetime | None = None
```

Changes from EP's schemas:
- Added `target` and `region` fields (EP only had price; now gen/load too)
- Added `forecast_lower`/`forecast_upper` for prediction intervals
- Added `ensemble_method` (blend vs stacking, per decision #3)
- Added `pi_coverage` metrics
- Removed EP's per-model error tracking from the API response (keep in MLflow)

---

### 1.10 Makefile

```makefile
.PHONY: help install lint format test mlflow serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with dev deps
	pip install -e ".[dev]"

lint:  ## Check formatting and lint
	ruff check energy_forecasting/ tests/
	ruff format --check energy_forecasting/ tests/

format:  ## Auto-format and fix
	ruff check --fix energy_forecasting/ tests/
	ruff format energy_forecasting/ tests/

test:  ## Run tests
	pytest tests/ -v

mlflow:  ## Start MLflow UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:  ## Start FastAPI dev server
	uvicorn energy_forecasting.api.app:app --reload --port 8000

clean:  ## Remove compiled files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# ── Stage 2+ targets (stubs, implemented in later stages) ───────────
# data:       ## Download all data from scratch
# update:     ## Incremental update all sources
# process:    ## Clean and merge -> processed/merged.parquet
# forecast:   ## Run daily inference
# retrain:    ## Full retrain pipeline
# sync:       ## Pull latest data from GitHub Release
```

---

### 1.11 .gitignore

```gitignore
# Data (large, downloaded at runtime)
data/

# Models (stored in GitHub Releases)
models/

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Environment
.env
.venv/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
*.Zone.Identifier

# Jupyter
.ipynb_checkpoints/

# Deploy data (generated, not source)
deploy/data/
```

---

### 1.12 Location Data Extraction

Extract EMA's `eu_locations.py` (1,817 lines of Python) to `data/locations/eu_locations.json`. This is pure data — coordinates, capacities, TSO mappings — so JSON is the right format (decision #13).

**Script (run once during stage 1):**

```python
"""Extract eu_locations.py to JSON. Run once: python scripts/extract_locations.py"""
import json, sys
sys.path.insert(0, "/path/to/energy_market_analysis")
from data_collection_modules.eu_locations import countries_metadata

# Serialize to JSON, converting any non-serializable types
with open("data/locations/eu_locations.json", "w") as f:
    json.dump(countries_metadata, f, indent=2, default=str)
```

Verify the JSON preserves all fields: `name`, `lat`, `lon`, `capacity_mw` (for generation sites), `population` (for cities), `TSO`, `suffix`, `type`, `available_targets`. Spot-check a few locations against the original Python file.

---

### 1.13 Skeleton Files

Create empty `__init__.py` files and stub modules so imports work and the package structure is established. Modules that are implemented in later stages get a docstring and nothing else:

```python
# energy_forecasting/data/sources.py
"""DataSource base class and implementations. See stage 2."""

# energy_forecasting/features/parser.py
"""Suffix DSL parser. See stage 4."""

# energy_forecasting/modeling/training.py
"""Training loop with MLflow integration. See stage 5."""
```

This ensures `pip install -e .` works and that imports like `from energy_forecasting.config.columns import SHORT_NAMES` are testable immediately.

---

### 1.14 Milestone & Tests

**Tests for stage 1** (`tests/`):

- **`test_io.py`** — `save_parquet` + `load_parquet` round-trip: create a DataFrame with float64/int64 columns, save with downcasting, reload, verify values match within float32 tolerance. Test that zstd compression produces a smaller file than uncompressed.
- **`test_columns.py`** — `SHORT_NAMES` has no duplicate values. `REVERSE_SHORT_NAMES` round-trips correctly. All `SMARD_FILTER_KEYS` produce valid column names via `clean_column_name()`. All short names used in `AVAILABILITY_RULES` exist in `SHORT_NAMES`.
- **`test_cleaning_rules.py`** — `clean()` runs without error on a synthetic DataFrame with the expected columns. Helper functions tested individually: `fill_zero_after` zeros the right rows, `clip_bounds` handles wildcards, `interpolate_gaps` respects `max_gap`, etc.
- **`test_availability.py`** — `AVAILABILITY_RULES` covers all short names that appear in the feature lists (once feature lists exist in stage 4; for now, test the rules parse correctly).
- **`test_schemas.py`** — Pydantic schemas can be instantiated with sample data. `ForecastResponse` serialises to JSON matching the expected dashboard format. `HourlyForecast` accepts `None` for interval bounds.

**Milestone checklist:**

- [ ] `pip install -e ".[dev]"` works on Python 3.13
- [ ] `make lint` and `make format` pass with zero issues
- [ ] `make test` passes all stage 1 tests
- [ ] `from energy_forecasting.config.columns import SHORT_NAMES` works
- [ ] `from energy_forecasting.config.cleaning import clean` works
- [ ] `from energy_forecasting.api.schemas import ForecastResponse` works
- [ ] `save_parquet` / `load_parquet` round-trip tested
- [ ] `data/locations/eu_locations.json` extracted and spot-checked
- [ ] All stub modules importable (no circular imports)

---
