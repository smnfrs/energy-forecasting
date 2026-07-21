## Stage 4: Feature Engineering

**Goal:** Feature lists for both price and gen/load models produce correct datasets. A suffix DSL makes feature definitions declarative and auditable; pure functions handle computation; leakage validation prevents information leakage; caching avoids recomputation.

**Source material:**
- EP: `src/features/transforms.py` (spreads, net exports, generation %, temporal, holidays), `src/features/ts_transforms.py` (rolling stats, EWMA, lags, daily aggregates), `src/config/features.py` (column lists, rolling specs, availability rules), `src/config/pipelines.py` (v5_slim/v5_full hourly ~line 1200+), `src/config/temporal.py` (German state populations, cyclical periods), `src/features/validation.py` (leakage validation)
- EMA: `data_modules/feature_eng.py` (WeatherWindPowerFE, WeatherSolarPowerFE, WeatherLoadFE, spatial aggregation, physics helpers, create_time_features), `data_collection_modules/eu_locations.py` (location metadata), `data_modules/data_classes.py` (HistForecastDataset, Fourier features, target lags)
- Already implemented (stages 1-3): `config/columns.py` (SHORT_NAMES registry, 85 entries), `config/availability.py` (19 AvailabilityRule entries), `data/weather.py` (VARS_BASIC, VARS_WIND, VARS_RADIATION), `data/locations/eu_locations.json` (76 locations)
- Stubs to fill: `features/parser.py`, `features/engine.py`, `features/market.py`, `features/weather_wind.py`, `features/weather_solar.py`, `features/weather_load.py`, `features/spatial.py`, `features/validation.py`

**Key design decisions (vs EP/EMA):**
- **No overwrite pattern.** EP replaced raw hourly columns with lagged daily means (e.g. `stromerzeugung_*` → D-2 mean) to prevent leakage by mutation. We don't do this. Instead, every feature in the list has an explicit suffix encoding its lag, and the validation rules catch mistakes. The engine computes features from raw data — no column mutation, no ordering dependencies, no confusion about what a bare column name means.
- **No sklearn transformers.** Pure functions only — extract the math, discard the ceremony.
- **Feature lists use short names exclusively.** The DSL short names (from `config/columns.py` SHORT_NAMES) make lists human-readable. Long column names are internal.
- **Weather FE computed once, reused across models.** Weather features are computed during gen/load model training (stage 5 Optuna), saved as parquet, then loaded as input columns for the price model. The engine doesn't need to know about weather — it just operates on whatever columns are in the input DataFrame.
- **Fourier features included** as a feature type (configurable period and order), not deferred to stage 5.

---

### 4.1 Config: Feature Constants

**`energy_forecasting/config/features.py`** (currently a stub) — Feature list definitions and computation constants. All feature lists use short names from the suffix DSL.

```python
"""Feature list definitions and computation constants.

Feature lists use the suffix DSL (see features/parser.py). Short names
resolve via config/columns.py SHORT_NAMES registry.
"""

from datetime import date

# ── Temporal encoding ────────────────────────────────────────────
CYCLICAL_PERIODS: dict[str, int] = {
    "hour": 24,
    "day_of_week": 7,
    "month": 12,
}

# German state populations (2023 estimates, millions).
# Population-weight holiday indicators: national holidays → 1.0,
# state-specific → fraction of population observing.
# Ported from EP's src/config/temporal.py.
GERMAN_STATE_POPULATIONS: dict[str, float] = {
    "NW": 17.9,  "BY": 13.2,  "BW": 11.1,  "NI": 8.0,
    "HE": 6.3,   "SN": 4.1,   "RP": 4.1,   "BE": 3.7,
    "SH": 2.9,   "BB": 2.5,   "ST": 2.2,   "TH": 2.1,
    "HH": 1.9,   "MV": 1.6,   "SL": 1.0,   "HB": 0.7,
}

# ── Trend epoch ──────────────────────────────────────────────────
DAY_INDEX_EPOCH = date(2015, 1, 5)  # matches EP's 2015-01-05 00:00 CET
YEAR_INDEX_BASE = 2015

# ── Generation columns (for percentage computation) ──────────────
GENERATION_COLUMNS: list[str] = [
    "stromerzeugung_biomasse",
    "stromerzeugung_braunkohle",
    "stromerzeugung_erdgas",
    "stromerzeugung_kernenergie",
    "stromerzeugung_photovoltaik",
    "stromerzeugung_pumpspeicher",
    "stromerzeugung_sonstige_erneuerbare",
    "stromerzeugung_sonstige_konventionelle",
    "stromerzeugung_steinkohle",
    "stromerzeugung_wasserkraft",
    "stromerzeugung_wind_offshore",
    "stromerzeugung_wind_onshore",
]

RENEWABLE_COLUMNS: list[str] = [
    "stromerzeugung_wind_onshore",
    "stromerzeugung_wind_offshore",
    "stromerzeugung_photovoltaik",
]

# ── Neighbour price columns (for spread computation) ─────────────
NEIGHBOUR_PRICES: list[str] = [
    "marktpreis_belgien",
    "marktpreis_daenemark_1",
    "marktpreis_daenemark_2",
    "marktpreis_frankreich",
    "marktpreis_italien_(nord)",
    "marktpreis_niederlande",
    "marktpreis_norwegen_2",
    "marktpreis_oesterreich",
    "marktpreis_polen",
    "marktpreis_schweden_4",
    "marktpreis_schweiz",
    "marktpreis_slowenien",
    "marktpreis_tschechien",
    "marktpreis_ungarn",
]

# ── Cross-border flow pairs (for net export computation) ─────────
# (export_column, import_column, country_short_name)
# Derived from CROSS_BORDER_DE_LU in config/columns.py.
FLOW_PAIRS: list[tuple[str, str, str]] = [
    # ... 14 tuples, populated during implementation from CROSS_BORDER_DE_LU
]

# ── Default Fourier config ───────────────────────────────────────
# period=24 captures daily seasonality; order=3 gives 3 sin/cos pairs.
# The period and order are tunable hyperparameters in stage 5.
DEFAULT_FOURIER_CONFIG: dict = {"period": 24, "order": 3}

# ── Feature lists (suffix DSL) ───────────────────────────────────
PRICE_FEATURES_SLIM: list[str] = [
    # ... ~84 features, defined in section 4.8 ...
]

PRICE_FEATURES_FULL: list[str] = [
    # ... ~138 features ...
]

GEN_WIND_ONSHORE_FEATURES: list[str] = [...]
GEN_WIND_OFFSHORE_FEATURES: list[str] = [...]
GEN_SOLAR_FEATURES: list[str] = [...]
LOAD_FEATURES: list[str] = [...]
```

**What's new vs EP/EMA:** EP spread constants across `config/features.py`, `config/temporal.py`, and `config/pipelines.py` (1,759 lines, mostly dead code). We consolidate into one file with declarative feature lists.

---

### 4.2 Config: Location Metadata Loader

**`energy_forecasting/config/locations.py`** (new file, ~60 lines) — Load and query `data/locations/eu_locations.json`.

```python
"""Location metadata for weather feature engineering.

Wraps eu_locations.json (ported from EMA's eu_locations.py) with
typed access. The JSON was converted from EMA's Python dict during
stage 2 data collection setup.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from energy_forecasting.config import LOCATIONS_DIR


class LocationMeta(TypedDict, total=False):
    name: str
    label: str
    type: str           # "city", "onshore wind farm", "offshore wind farm", "solar farm"
    suffix: str         # e.g. "_woff_enbw", "_city_berlin"
    TSO: str
    lat: float
    lon: float
    capacity: float     # MW (wind/solar farms)
    n_turbines: int     # wind farms
    n_panels: int       # solar farms
    population: int     # cities
    total_energy_consumption: float  # GWh/year (cities)


@lru_cache
def load_locations() -> list[LocationMeta]:
    """Load all locations from eu_locations.json."""
    ...


def locations_for_tso(tso: str, asset_type: str) -> list[LocationMeta]:
    """Filter locations by TSO and asset type.

    asset_type: "offshore", "onshore", "solar", "cities"
    Maps to location type strings in the JSON.
    """
    ...
```

---

### 4.3 Suffix DSL Parser

**`energy_forecasting/features/parser.py`** (~150 lines) — Parses feature strings into structured specs.

#### Full grammar specification

```
feature      := interaction | simple
interaction  := simple "__x__" simple
simple       := short_name suffix?
suffix       := ewma_suffix | lag_suffix | agg_suffix | fourier_suffix | daily_agg_suffix
lag_suffix   := "_h" INT                              # hourly lag (shift)
agg_suffix   := "_d" INT ("_d" INT)?                  # day range
                ("_eh" INT)?                           # optional end-hour on final day
                ("_h" INT "_h" INT)?                   # optional hour filter (all days)
                ("_" STAT)?                            # optional statistic
ewma_suffix  := "_ewma_" INT ("_d" INT ("_h" INT)?)?  # EWMA with optional cutoff
fourier_suffix := "_fourier_" INT "_" INT             # period_order
daily_agg_suffix := "_daily_" STAT                    # broadcast daily aggregate

STAT := "avg" | "std" | "min" | "max" | "sum" | "range"
INT  := [0-9]+
```

**Semantics of each suffix type:**

| Suffix | Example | Meaning |
|--------|---------|---------|
| (none) | `prog_load` | Raw column value. Must have `max_offset >= 0` in availability rules (forecasts, temporal, static) |
| `_h N` | `price_h24` | Value N hours ago: `series.shift(N)`. Same-hour-yesterday = `_h24`, same-hour-last-week = `_h168` |
| `_d X` | `price_d7` | Single-day agg: all hours of day D-7, default stat=avg |
| `_d X_d Y` | `price_d7_d1` | Multi-day agg: all hours from D-7 through D-1, default stat=avg |
| `_d X_d Y_STAT` | `price_d7_d1_std` | Multi-day agg with explicit stat |
| `_d X_d Y_eh H` | `price_d7_d1_eh8` | Multi-day agg, end-hour cutoff: D-7 00:00 through D-1 08:00 |
| `_d X_d Y_h A_h B` | `price_d7_d1_h8_h19` | Multi-day agg, hour filter: only hours [A, B) on every day |
| `_d X_d Y_h A_h B_STAT` | `price_d7_d1_h8_h19_avg` | Fully specified with hour filter + stat |
| `_d X_d Y_range` | `price_d7_d1_range` | Computes max - min over the window |
| `_ewma_S` | `price_ewma_6` | EWMA with span S hours, no cutoff (uses all history) |
| `_ewma_S_d D` | `price_ewma_6_d1` | EWMA cutoff at end of day D-1 |
| `_ewma_S_d D_h H` | `price_ewma_6_d1_h10` | EWMA cutoff at hour H on day D-1 |
| `_fourier_P_O` | `hour_fourier_24_3` | Fourier with period P, order O (produces 2×O columns) |
| `_daily_STAT` | `prog_gen_wind_pv_daily_max` | Group by date, compute STAT, broadcast to all 24 hours |
| `__x__` | `ttf_ewma_720_d2__x__day_index` | Interaction (element-wise product of left × right) |

**End-hour (`_eh`) vs hour-filter (`_h_h`) — when to use which:**
- `_eh H` truncates the final day at hour H. All preceding days in the range are complete (hours 0-23). Use for "all data up to a cutoff point," e.g. morning actuals: `residual_load_d1_eh10` = D-1 hours 0-9.
- `_h A_h B` filters to hours [A, B) on *every* day in the range. Use for "peak hours only across a multi-day window," e.g. `price_d7_d1_h8_h19` = hours 8-18 on each of the 7 days.
- `_eh` and `_h_h` are mutually exclusive — a feature string cannot have both.

**Parsing notes:**
- Short names resolved via longest-prefix match in `SHORT_NAMES`. Unknown names → `ValueError` with `difflib.get_close_matches` suggestion.
- `_d` values in the feature string are always positive integers representing days back. Internally stored as negative (`_d7` → `start_day=-7`).
- `_d X` alone (single day) is shorthand for `_d X_d X`.
- `_range` is a stat type that computes `max - min` over the window.
- `_daily_` distinguishes broadcast-to-hourly aggregation from historical rolling. `_d0` would also work but `_daily_` is clearer for forecast columns.
- Fourier produces multiple output columns. The parser returns a single `FeatureSpec` but the engine expands it into 2×order columns.

#### Parser output types

```python
@dataclass(frozen=True)
class HourlyLag:
    hours: int

@dataclass(frozen=True)
class Aggregation:
    start_day: int    # negative, e.g. -7
    end_day: int      # negative, e.g. -1
    stat: str         # "avg", "std", "min", "max", "sum", "range"
    end_hour: int | None = None   # truncate final day at this hour
    hour_start: int | None = None # filter all days to [hour_start, hour_end)
    hour_end: int | None = None

@dataclass(frozen=True)
class EWMA:
    span: int
    cutoff_day: int | None = None
    cutoff_hour: int | None = None

@dataclass(frozen=True)
class Fourier:
    period: int
    order: int

@dataclass(frozen=True)
class DailyAggregate:
    stat: str  # "sum", "mean", "max", "min", "std"

@dataclass(frozen=True)
class FeatureSpec:
    base: str       # short name
    raw_col: str    # resolved column from SHORT_NAMES
    lag: HourlyLag | None = None
    agg: Aggregation | None = None
    ewma: EWMA | None = None
    fourier: Fourier | None = None
    daily_agg: DailyAggregate | None = None

@dataclass(frozen=True)
class InteractionSpec:
    left: FeatureSpec
    right: FeatureSpec
```

These are pure data containers — no computation methods. They bridge parsing and the engine.

---

### 4.4 Market Feature Functions

**`energy_forecasting/features/market.py`** (~300 lines) — Pure functions that compute market-derived features. Each function takes a DataFrame (or Series) and parameters, returns computed columns.

#### 4.4.1 Price spreads

```python
def compute_price_spreads(
    df: pd.DataFrame, neighbours: list[str] | None = None,
) -> pd.DataFrame:
    """spread_{country} = target_price - neighbour_price.

    Positive spread = DE-LU more expensive than neighbour.
    Ported from: EP's PriceSpreadTransformer.
    """
```

#### 4.4.2 Net exports

```python
def compute_net_exports(
    df: pd.DataFrame, flow_pairs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """net_export_{country} = exports - imports.

    Also computes total_exports, total_imports (sum across countries).
    Ported from: EP's NetExportTransformer + v5 total_flows.
    """
```

#### 4.4.3 Generation percentages

```python
def compute_generation_pct(
    df: pd.DataFrame,
    sources: list[str] | None = None,
    add_renewable_pct: bool = False,
    add_supply_demand_gap: bool = False,
    add_prognosticated_pct: bool = False,
) -> pd.DataFrame:
    """pct_{source} = source / total_generation.

    Optionally: pct_renewable, supply_demand_gap, total_generation,
    pct_prog_sonstige, pct_prog_wind_und_photovoltaik.
    Ported from: EP's GenerationPercentageTransformer + PrognosticatedPercentageTransformer.
    """
```

#### 4.4.4 Rolling statistics

```python
def compute_rolling_stat(
    series: pd.Series,
    start_day: int,
    end_day: int,
    stat: str = "avg",
    end_hour: int | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
) -> pd.Series:
    """Rolling statistic over a day-relative historical window.

    For each row at hour H on day D:
    - Window = all rows from day D+start_day through D+end_day
      (start_day and end_day are negative, e.g. -7, -1)
    - end_hour: truncate the final day at this hour (e.g. end_hour=8 →
      include hours 0-7 on the last day, all hours on preceding days).
      Use for "all data up to a cutoff point."
    - hour_start/hour_end: filter to hours [hour_start, hour_end) on
      *every* day in the range. Use for "peak hours only."
    - end_hour and hour_start/hour_end are mutually exclusive.
    - stat: "avg", "std", "min", "max", "sum", "range" (range = max - min)

    The merged dataset uses tz-naive local delivery hours (Europe/Berlin),
    so hour-of-day filtering works directly on index.hour.

    Also handles daily aggregate + broadcast (start_day=0, end_day=0):
    groups by date, computes stat, broadcasts to all 24 hours. This
    unifies EP's RollingStatsTransformer and HourlyDailyAggregateTransformer
    — the only difference is whether the window is historical or current-day.

    Ported from: EP's RollingStatsTransformer and HourlyDailyAggregateTransformer.
    """
```

**Unification note:** `compute_daily_aggregates` (from the original plan) is just `compute_rolling_stat` with `start_day=0, end_day=0`. We use one function. The `_daily_max` suffix parses to `Aggregation(start_day=0, end_day=0, stat="max")` internally.

**Share feature:** EP's `HourlyDailyAggregateTransformer` also computed `{col}_share = col / daily_sum`. This is encoded as a separate feature in the list (e.g. `prog_gen_wind_pv_daily_share`), where `_daily_share` maps to a special stat.

#### 4.4.5 EWMA with information cutoff

```python
def compute_ewma(
    series: pd.Series,
    span: int,
    cutoff_day: int | None = None,
    cutoff_hour: int | None = None,
) -> pd.Series:
    """EWMA with information cutoff boundary.

    - cutoff_day=-1 → use data up to end of yesterday
    - cutoff_day=-1, cutoff_hour=10 → use data up to yesterday 10:00
    - cutoff_day=-2 → use data up to end of two days ago

    EWMA value at the cutoff is broadcast to all hours of the prediction
    day (constant within-day).

    Since the index is tz-naive Europe/Berlin, cutoff_hour is local time.

    Spans from EP v5: 6, 24, 168, 720, 2160 hours
    (~¼d, 1d, 1w, 1m, 3m half-life).

    Ported from: EP's EWMATransformer.
    """
```

#### 4.4.6 Hourly lag

```python
def compute_hourly_lag(series: pd.Series, hours: int) -> pd.Series:
    """Value N hours back: series.shift(hours).

    Handles all lag types:
    - Same-hour yesterday: hours=24
    - Same-hour last week: hours=168
    - Adjacent-hour (H-1): hours=25 (i.e. hour before, previous day)

    Ported from: EP's SameHourLagTransformer + adjacent-hour lags.
    """
```

One function replaces both `compute_same_hour_lag` and `compute_hourly_lag` from the original plan. `shift(N)` is `shift(N)` regardless of whether N is a multiple of 24.

#### 4.4.7 Temporal features

```python
def compute_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar and cyclical time features.

    Returns: hour_of_day, day_of_week, day_of_month, month, week_of_year,
    is_weekend, hour_sin/cos (period 24), day_of_week_sin/cos (period 7),
    month_sin/cos (period 12).

    Ported from: EP's TemporalFeatureTransformer + EMA's create_time_features.
    """


def compute_german_holidays(index: pd.DatetimeIndex) -> pd.Series:
    """Population-weighted German holiday indicator (0.0 to 1.0).

    Uses the `holidays` library for all 16 states. National holidays → 1.0,
    state-specific → fraction of population observing.

    Ported from: EP's GermanHolidayTransformer.
    """
```

#### 4.4.8 Fourier features

```python
def compute_fourier_features(
    index: pd.DatetimeIndex, period: int, order: int,
) -> pd.DataFrame:
    """Deterministic Fourier terms for arbitrary-period seasonality.

    For order=O, produces 2*O columns:
      sin_1, cos_1, sin_2, cos_2, ..., sin_O, cos_O
    where sin_k = sin(2π * k * t / period), cos_k = cos(2π * k * t / period),
    and t is the integer position in the index.

    period=24, order=3 captures daily seasonality with 3 harmonics.
    period=168, order=2 captures weekly seasonality with 2 harmonics.

    Ported from: EMA's statsmodels DeterministicProcess + Fourier usage.
    Reimplemented directly (numpy sin/cos) — no statsmodels dependency needed.
    """
```

#### 4.4.9 Trend and interaction features

```python
def compute_day_index(index: pd.DatetimeIndex) -> pd.Series:
    """Days since DAY_INDEX_EPOCH (2015-01-05). Integer."""


def compute_interaction(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise product: left * right."""
```

---

### 4.5 Weather Feature Classes

Port EMA's three `WeatherBasedFE` subclasses. Each class:
- Takes a config dict (all features togglable)
- Has `transform(df) -> df` that computes physics features per location
- Has `suggest_optuna(trial) -> config_dict` for hyperparameter search (stage 5)
- Delegates spatial aggregation to `features/spatial.py`

**Compute-once, reuse-across-models pattern:**
1. During gen/load model Optuna training (stage 5), weather FE runs with the trial's config → saves output to `data/processed/weather_features/{target}_{tso}.parquet`
2. For price models, the saved weather feature outputs are loaded as additional input columns — the engine doesn't recompute them
3. This means weather FE classes are called by the model training loop (stage 5), not by `engineer_features()` directly. The engine just sees the weather columns as regular input columns.

#### 4.5.1 Spatial aggregation

**`energy_forecasting/features/spatial.py`** (~120 lines)

```python
"""Spatial aggregation for multi-location weather data.

Collapses per-location features (temperature_2m_woff_enbw, temperature_2m_woff_borkum)
into aggregated columns (temperature_2m_agg).

Ported from: EMA's WeatherBasedFE._apply_spatial_aggregation().
"""


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. R=6371."""


def aggregate_locations(
    df: pd.DataFrame,
    locations: list[LocationMeta],
    method: str,
) -> pd.DataFrame:
    """Collapse per-location columns into aggregated features.

    Methods:
      mean, max, idw (1/d² from centroid), capacity, n_turbines, n_panels,
      population, energy, distance_capacity, distance_n_turbines,
      distance_population, distance_energy
    """
```

#### 4.5.2 Shared physics

**`energy_forecasting/features/weather_physics.py`** (~80 lines) — Shared formulas used by all three weather FE classes.

```python
def compute_air_density(pressure, temperature):
    """Dry-air ideal gas: ρ = P×100 / (287.05 × (T+273.15))"""

def compute_air_density_moist(temperature, pressure, humidity):
    """Moist-air density with Magnus vapour pressure correction."""

def compute_dew_point_temperature(temperature, humidity):
    """Magnus formula: a=17.62, b=243.12."""

def compute_vapor_pressure(temperature, humidity):
    """6.112 × exp(17.67×T / (T+243.5)) × (RH/100)"""

def compute_wind_power_density(wind_speed, air_density):
    """WPD = 0.5 × ρ × (ws/3.6)³ (wind_speed in km/h)"""
```

Wind-only functions (`compute_wind_shear`, `compute_turbulence_intensity`, `compute_gust_factor`) in `weather_wind.py`. Load-only (`compute_wind_chill`, `compute_humidex`) in `weather_load.py`.

#### 4.5.3 Wind power features

**`energy_forecasting/features/weather_wind.py`** (~200 lines)

```python
class WeatherWindPowerFE:
    """Wind power feature engineering.

    Input raw columns per location suffix:
        temperature_2m, relative_humidity_2m, surface_pressure, precipitation,
        cloud_cover, wind_speed_10m, wind_speed_100m, wind_direction_10m,
        wind_direction_100m, wind_gusts_10m

    Computed features (all conditional on config):
        air_density, wind_power_density, air_density_moist,
        wind_dir_sin/cos, wind_shear, turbulence_intensity,
        wind_ramp, gust_factor, dew_point_temperature, vapor_pressure

    Lags (wind speed): none / small [1,6]h / large [1,6,12]h
    Lags (precipitation): none / small [1,6]h / large [1,6,12,24]h

    Spatial agg: mean, max, idw, capacity, n_turbines,
                 distance_capacity, distance_n_turbines
    """

    def __init__(self, config: dict): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def suggest_optuna(trial, fixed: dict | None = None) -> dict: ...
```

#### 4.5.4 Solar power features

**`energy_forecasting/features/weather_solar.py`** (~180 lines)

```python
class WeatherSolarPowerFE:
    """Solar power feature engineering.

    Additional raw columns: shortwave_radiation, direct_radiation,
    diffuse_radiation, direct_normal_irradiance, global_tilted_irradiance,
    terrestrial_radiation. solar_elevation_deg, solar_azimuth_deg already
    computed during stage 2 collection (pysolar).

    Computed features: cloud_cover_fraction, clear_sky_fraction,
    air_density, air_density_moist, direct_ratio, diffuse_ratio,
    dni_ratio, global_tilted_ratio, dew_point_temperature, vapor_pressure

    Lags (precip): none / small [1,6]h / large [1,6,12,24]h
    Lags (cloud): none / small [1,3]h / medium [1,3,6]h / large [1,3,6,12]h
    Lags (shortwave): none / small [1,3]h / medium [1,3,6]h / large [1,3,6,12]h

    Spatial agg: mean, max, idw, capacity, n_panels,
                 distance_capacity, distance_n_panels
    """

    def __init__(self, config: dict): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def suggest_optuna(trial, fixed: dict | None = None) -> dict: ...
```

#### 4.5.5 Load features

**`energy_forecasting/features/weather_load.py`** (~200 lines)

```python
class WeatherLoadFE:
    """Load (demand) feature engineering.

    Uses city weather data (VARS_BASIC + VARS_WIND + VARS_RADIATION).

    Computed features: HDH, CDH, dew_point_temperature, dew_point_spread,
    temp_gradient, wind_chill, humidex, pressure_trend, air_density,
    rain_indicator, wind_speed_gradient, wind_u, wind_v,
    wind_power_density, cloud_cover_fraction, effective_solar

    Lags (temperature): none / small [1,3]h / medium [1,3,6]h
    Lags (precip): none / small [1,3]h / medium [1,3,6]h
    Lags (cloud): none / small [1,3]h / medium [1,3,6]h
    Rolling (temperature): none / short [3,6]h / long [6,12,24]h

    Spatial agg: mean, max, idw, population, energy,
                 distance_population, distance_energy
    """

    def __init__(self, config: dict): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def suggest_optuna(trial, fixed: dict | None = None) -> dict: ...
```

---

### 4.6 Feature Computation Engine

**`energy_forecasting/features/engine.py`** (~200 lines) — Orchestrates feature computation from a feature list.

```python
def engineer_features(
    df: pd.DataFrame,
    feature_list: list[str],
    validate: bool = True,
) -> pd.DataFrame:
    """Compute all features in feature_list from the input DataFrame.

    Steps:
    1. Parse all feature strings → list of FeatureSpec/InteractionSpec
    2. Validate against availability rules (if validate=True)
    3. Identify required base columns (from SHORT_NAMES)
    4. Compute derived base columns on demand:
       - Price spreads (if any spread_* base in list)
       - Net exports (if any net_export_* base in list)
       - Generation percentages (if any gen_pct_* base in list)
       - Temporal features (if any temporal base in list)
       - Fourier features (if any fourier suffix in list)
    5. Compute feature-specific operations:
       - Hourly lags (shift)
       - Rolling aggregations (historical windows)
       - Daily aggregates (current-day, broadcast)
       - EWMA with cutoff
    6. Compute interactions (left * right)
    7. Select exactly the requested columns

    The engine works on raw data — no column overwriting or mutation.
    Every feature's lag is explicit in its suffix. Validation (step 2)
    ensures no leaked data makes it into the output.

    Args:
        df: Input DataFrame with raw columns. For price models, this is
            merged.parquet (optionally with weather feature columns pre-joined).
            For gen/load models, this is per-TSO SMARD + weather FE output.
        feature_list: List of suffix DSL feature strings (short names).
        validate: Run leakage validation before computing (default True).

    Returns:
        DataFrame with exactly the requested columns, plus the index.
    """
```

**No overwrite pattern.** The engine reads raw hourly values, computes what's requested (e.g. `gen_wind_on_d2` = D-2 daily mean of wind onshore generation), and returns it as a new column. The original `stromerzeugung_wind_onshore` column is never mutated. If someone accidentally puts a bare actual column (e.g. `gen_wind_on` with no suffix) in the feature list, the validation catches it because `stromerzeugung_wind_onshore` has `max_offset_days=-1` in the availability rules — bare columns are only allowed for forecasts and deterministic features.

**Incremental extension.** For extending an existing feature dataset with new data:

```python
def extend_features(
    existing_df: pd.DataFrame,
    new_raw_df: pd.DataFrame,
    feature_list: list[str],
    context_days: int = 30,
) -> pd.DataFrame:
    """Extend an existing feature dataset with newly available raw data.

    Computes features only for timestamps in new_raw_df, but uses
    context_days of overlap from existing raw data for rolling/EWMA
    context. Returns the concatenation of existing_df + new features.

    Use case: MLflow stores a feature dataset covering Jan 2015 - Feb 2026.
    New raw data for March 2026 arrives. This function computes features
    for March using Feb as context, and appends.

    Args:
        existing_df: Previously computed feature dataset.
        new_raw_df: New raw data (merged.parquet format) to compute features for.
        feature_list: Same feature list used for the existing dataset.
        context_days: Days of overlap to include for rolling/EWMA context.

    Returns:
        Concatenated DataFrame: existing_df + newly computed features.
    """
```

Persistence is handled by MLflow (stage 5): feature datasets are logged as artifacts, and `extend_features` loads the previous version from MLflow to avoid recomputing from scratch. No local caching layer needed — MLflow with a local SQLite backend reads from the same disk.

---

### 4.7 Leakage Validation

**`energy_forecasting/features/validation.py`** (~100 lines) — Declarative check that feature suffixes respect availability rules.

```python
def validate_feature_list(
    feature_list: list[str],
    rules: list[AvailabilityRule] | None = None,
) -> list[str]:
    """Check all features for information leakage.

    Returns a list of violation descriptions (empty = all valid).

    Rules per suffix type:
    - No suffix (bare name): max_offset_days must be >= 0
      (only forecasts, temporal, static columns allowed bare)
    - Hourly lag (_h N): N/24 must be >= |max_offset_days|.
      Also checks cutoff_hour if applicable.
    - Aggregation (_d X ...): start_day must be <= max_offset_days.
      If end_day == max_offset_days and rule has cutoff_hour,
      the aggregation window must not extend past cutoff.
    - EWMA (_ewma_S_d D_h H): cutoff_day must be <= max_offset_days.
      If same day, cutoff_hour must be <= rule cutoff_hour.
    - Daily aggregate (_daily_STAT): only valid for forecast columns
      (max_offset_days >= 0) since it aggregates current-day values.
    - Fourier: always valid (deterministic function of timestamp).
    - Interaction: both sides validated independently.
    """
```

**What's new vs EP:** EP's validator walked sklearn pipeline steps to infer what each step consumed, needed a complex `_check_unhandled()` to find unprotected columns. Our validator works on parsed specs directly — the feature list *is* the specification.

---

### 4.8 Feature Lists

**In `config/features.py`** — declarative feature specifications using the suffix DSL. All use short names.

#### 4.8.1 Price features (slim, ~84 features)

Matching EP's `preprocessor_v5_slim_hourly(morning_cutoff_cet=10)`. Organised by category:

| Category | Count | Examples (short name DSL) |
|----------|-------|--------------------------|
| Target rolling stats | 12 | `price_d7`, `price_d7_d1`, `price_d7_d1_std`, `price_d7_d1_max`, `price_d7_d1_min`, `price_d7_d1_h8_h19`, `price_d30_d1`, `price_d30_d1_std`, `price_d2_d1_std`, `price_d3_d1_std`, `price_d30_d1_min`, `price_d30_d1_max` |
| Price ranges | 2 | `price_d7_d1_range`, `price_d30_d1_range` |
| Price EWMA (end-of-D-1) | 3 | `price_ewma_6_d1`, `price_ewma_24_d1`, `price_ewma_2160_d1` |
| Price EWMA (h10 cutoff) | 3 | `price_ewma_6_d1_h10`, `price_ewma_24_d1_h10`, `price_ewma_2160_d1_h10` |
| France EWMA | 3 | `price_fr_ewma_6_d1`, `price_fr_ewma_24_d1`, `price_fr_ewma_2160_d1` |
| Actual EWMA (h10) | 9 | `residual_load_ewma_{24,168,2160}_d1_h10`, `gen_wind_on_ewma_{...}`, `gen_solar_ewma_{...}` |
| Commodity EWMA (D-2) | 5 | `carbon_ewma_24_d2`, `ttf_ewma_24_d2`, `ttf_ewma_720_d2`, ... |
| Morning actuals | 3 | `residual_load_d1_eh10`, `gen_wind_on_d1_eh10`, `gen_wind_off_d1_eh10` |
| Same-hour lags (price) | 4 | `price_h24`, `price_h48`, `price_h168`, `price_h336` |
| Same-hour lags (neighbours) | 2 | `price_fr_h24`, `price_ch_h24` |
| Same-hour lags (gen) | 3 | `gen_wind_on_h48`, `gen_wind_off_h48`, `gen_solar_h48` |
| Actuals (D-2 daily mean) | ~6 | `gen_wind_on_d2`, `gen_wind_off_d2`, `gen_solar_d2`, `gen_gas_d2`, `load_d2`, ... |
| Neighbour prices (D-1 mean) | ~4 | `price_fr_d1`, `price_ch_d1`, `price_nl_d1`, `price_at_d1` |
| Commodities (D-2 mean) | 3 | `carbon_d2`, `ttf_d2`, `brent_d2` |
| Cross-border (D-2 mean) | 2 | `total_exports_d2`, `total_imports_d2` |
| Forecasts (no lag) | 4 | `prog_load`, `prog_gen_wind_pv`, `prog_residual`, `prog_gen_other` |
| Forecast daily aggregates | 2 | `prog_gen_wind_pv_daily_max`, `prog_load_daily_max` |
| Generation % | ~4 | `gen_pct_gas`, `gen_pct_hydro`, `gen_pct_pumped`, `gen_pct_other_renew`, `supply_demand_gap` |
| Prognosticated % | 2 | `pct_prog_other`, `pct_prog_wind_pv` |
| Temporal | ~5 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_holiday` |
| Trend + interactions | 7 | `day_index`, `year_index`, 5 interaction terms |

The exact list will be derived during implementation by comparing against EP's v5_slim pipeline output.

**Note:** Every feature that references actual data has an explicit lag suffix — `gen_wind_on_d2` clearly means "D-2 daily average of wind onshore generation." No bare column names for actual data.

#### 4.8.2 Price features (full, ~138 features)

Adds to slim: D-1 intra-day stats (`price_d1_max`, `price_d1_min`, `price_d1_std`), more morning actuals, neighbour price daily stats, extended same-hour lags (D-2), adjacent-hour lags (`price_h25`, `price_h26`), more daily aggregates (5-stat), `pct_renewable`, extended interaction terms (13 total).

#### 4.8.3 Gen/load features

Primarily weather features + time features + target lags. Weather columns come from the weather FE classes (section 4.5). The exact feature set depends on the weather FE config (Optuna hyperparameter in stage 5). Stage 4 defines default configs:

- **Wind onshore/offshore:** weather wind FE output + `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_holiday`
- **Solar:** weather solar FE output + temporal features
- **Load:** weather load FE output + temporal features + cross-target exogenous (wind/solar forecasts from other models)

Target lags (`{target}_lag_1` through `{target}_lag_N`) are coupled with recursive forecasting and handled in stage 5.

---

### 4.9 CLI Integration

**`energy_forecasting/cli.py`** — Add `features` command.

```python
@cli.command()
@click.option("--feature-set", type=click.Choice(["slim", "full"]), default="slim")
def features(feature_set: str):
    """Compute feature dataset from merged.parquet.

    Produces data/processed/features_{slim|full}.parquet.
    """
```

Also add a `features` target to the Makefile.

---

### 4.10 Tests

**`tests/test_parser.py`** (~35 tests) — DSL parser:
- Bare name resolution: `"price"` → `FeatureSpec(base="price", raw_col="target_price")`
- Hourly lag: `"price_h24"`, `"price_h168"`
- Single-day aggregation: `"price_d7"` → `Aggregation(start=-7, end=-7, stat="avg")`
- Multi-day aggregation: `"price_d7_d1"`, `"price_d7_d1_std"`, `"price_d7_d1_range"`
- Hour-filtered: `"price_d7_d1_h8_h19"`, `"price_d7_d1_h8_h19_avg"`
- End-hour cutoff: `"residual_load_d1_eh10"` → `Aggregation(start=-1, end=-1, end_hour=10, stat="avg")`
- End-hour multi-day: `"price_d7_d1_eh8"` → `Aggregation(start=-7, end=-1, end_hour=8, stat="avg")`
- End-hour + stat: `"price_d7_d1_eh8_std"` → `Aggregation(start=-7, end=-1, end_hour=8, stat="std")`
- End-hour and hour-filter mutually exclusive: `"price_d7_d1_eh8_h0_h10"` → ValueError
- EWMA variants: `"price_ewma_6"`, `"price_ewma_6_d1"`, `"price_ewma_6_d1_h10"`
- Fourier: `"hour_fourier_24_3"` → `Fourier(period=24, order=3)`
- Daily aggregate: `"prog_gen_wind_pv_daily_max"` → `DailyAggregate(stat="max")`
- Interaction: `"gen_solar_h48__x__day_index"`
- Unknown name → ValueError with "did you mean?"
- Invalid suffix → ValueError
- Round-trip: every feature in `PRICE_FEATURES_SLIM` parses without error

**`tests/test_market.py`** (~20 tests) — Market feature functions:
- Spread: correct sign, NaN propagation
- Net export: correct difference, totals
- Generation %: sums to ~1.0, supply_demand_gap, prognosticated pct
- Rolling stats: correct window, hour filtering, range stat = max - min
- Rolling with end_hour: `start_day=-7, end_day=-1, end_hour=8` includes all of D-7:D-2, hours 0-7 of D-1
- Rolling with `start_day=0, end_day=0`: broadcasts daily aggregate
- EWMA: cutoff boundary, value broadcast, different spans
- Hourly lag: shift(24) matches shift(168)/7 for stable series
- Temporal: cyclical in [-1,1], holiday weighting, day_index monotonic
- Fourier: correct number of output columns, values in [-1,1]

**`tests/test_weather.py`** (~15 tests) — Weather FE classes + physics:
- Physics: air_density > 0, WPD ≥ 0, wind_shear finite, dew_point < temperature
- Wind: all computed columns present for each config flag
- Solar: radiation ratios in [0,1], solar geometry columns present/absent per config
- Load: HDH ≥ 0, CDH ≥ 0, wind_chill ≤ temperature, effective_solar ≥ 0
- Spatial: mean of identical cols = original, IDW weights sum to 1, capacity-weighted
- Optuna: `suggest_optuna` returns valid config, all expected keys present

**`tests/test_validation.py`** (~25 tests) — Leakage validation, **extensive known-bad features**:
- **All PRICE_FEATURES_SLIM pass** validation
- **Bare actuals caught:** `"gen_wind_on"` (no suffix, max_offset=-1)
- **Bare prices caught:** `"price"` (no suffix, max_offset=-1)
- **Bare commodities caught:** `"carbon"` (no suffix, max_offset=-2)
- **Insufficient hourly lag — actuals:** `"gen_wind_on_h12"` (12h < 24h required), `"load_h20"` (20h < 24h)
- **Insufficient hourly lag — prices:** `"price_h12"` (needs ≥24h), `"price_fr_h6"` (needs ≥24h)
- **Insufficient hourly lag — commodities:** `"carbon_h24"` (needs ≥48h), `"ttf_h36"` (needs ≥48h)
- **Boundary hourly lag — actuals with cutoff:** `"gen_wind_on_h24"` should pass (exactly D-1 at same hour), `"gen_solar_h23"` should fail (23h < 24h)
- **Insufficient day lag — actuals:** `"gen_wind_on_d0"` (same-day actual), `"load_d0_eh10"` (same-day)
- **Insufficient day lag — prices:** `"price_d0"` (same-day price)
- **Insufficient day lag — commodities:** `"carbon_d1"` (D-1 but needs D-2), `"ttf_d1"` (D-1 but needs D-2)
- **EWMA insufficient cutoff:** `"price_ewma_24_d0"` (cutoff today), `"carbon_ewma_24_d1"` (D-1 but needs D-2)
- **EWMA hour cutoff violation:** `"gen_wind_on_ewma_24_d1_h12"` (cutoff h12 > h10 required)
- **Daily aggregate on actuals caught:** `"gen_wind_on_daily_max"` (same-day actual)
- **Daily aggregate on forecasts pass:** `"prog_load_daily_max"` (forecast, max_offset=0)
- **End-hour on leaked data:** `"gen_wind_on_d0_eh10"` (same-day actual, even with end-hour)
- **End-hour valid:** `"gen_wind_on_d1_eh10"` passes (D-1 with cutoff_hour=10, exactly at boundary)
- **End-hour exceeds cutoff:** `"gen_wind_on_d1_eh12"` fails (cutoff h12 > h10 required)
- **Interaction — one side leaked:** `"gen_wind_on__x__day_index"` (bare actual)
- **Interaction — both sides valid:** `"ttf_ewma_720_d2__x__day_index"` passes
- **Fourier always passes:** `"hour_fourier_24_3"` (deterministic)
- **Temporal always passes:** `"hour_sin"`, `"is_holiday"`, `"day_index"`
- **Edge case — exactly at boundary:** `"price_d1"` (D-1 price, exactly at offset), `"carbon_d2"` (exactly at D-2)

**`tests/test_engine.py`** (~10 tests) — Engine integration:
- Small feature list on synthetic data → correct shape and column names
- Engine with `validate=True` rejects leaked features
- Derived columns computed on demand (spreads only when spread features requested)
- Incremental extension produces correct output
- Unknown feature raises ValueError at parse time

### Implementation Order

1. **Config** (`config/features.py`, `config/locations.py`) — constants and metadata loader
2. **Parser** (`features/parser.py`) + tests — can be developed and tested in isolation
3. **Physics helpers** (`features/weather_physics.py`) — shared formulas, easy to unit test
4. **Market functions** (`features/market.py`) + tests — each function is independent
5. **Weather FE classes** (`features/weather_wind.py`, `weather_solar.py`, `weather_load.py`) + spatial + tests
6. **Validation** (`features/validation.py`) + tests — needs parser + availability rules
7. **Engine** (`features/engine.py`) + tests — ties parsing + market + validation
8. **Feature lists** (populate `PRICE_FEATURES_SLIM`, `PRICE_FEATURES_FULL`) — validate against EP v5
9. **CLI + Makefile** integration

Steps 1-4 are independent and can be parallelised. Steps 5-6 depend on 3. Steps 7-9 are sequential.

---

### What's NOT in Stage 4

- **Target transforms and scaling** — Stage 5 (`TransformedTargetRegressor`, feature scaling as model pipeline step)
- **Target lags for gen/load** — Stage 5 (coupled with recursive forecasting loop)
- **Feature selection** — Stage 5 (modelling decision, not engineering)
- **Cross-border flow columns for IT_N, HU, SI** — Computable if requested but excluded from default slim list (66% NaN, DE-AT-LU era only)

---

### Remaining Work (was blocked on weather data)

Weather downloads complete (2026-04-03). Validation results:
- [x] Validate `WeatherSolarPowerFE` on real solar weather data — PASS (all 5 TSOs, ~98.5K rows each, 18 cols, cloud fraction/radiation ratios within bounds)
- [x] Validate `WeatherLoadFE` on real city weather data — PASS (all 5 TSOs, ~98.5K rows each, 24 cols, HDH/CDH non-negative, per-city rain_indicator binary, aggregated rain_indicator is population-weighted fraction = expected)
- [x] Validate `WeatherWindPowerFE` on real data — PASS (offshore 2 TSOs + onshore 4 TSOs, WPD non-negative, sin/cos in [-1,1], no NaN outside lag warm-up)
- [ ] Integration test: full gen/load feature pipeline with real weather + per-TSO SMARD — deferred to stage 5 (requires training loop)
- [ ] Compare weather FE output against EMA's output for the same locations — deferred to stage 5
- [ ] Populate `GEN_SOLAR_FEATURES` and `LOAD_FEATURES` lists — deferred to stage 5 (feature lists depend on Optuna weather FE config)

---

### Outputs

```
data/processed/
  features_slim.parquet    ← ~84 columns, national hourly, tz-naive
  features_full.parquet    ← ~138 columns, national hourly, tz-naive
```

Per-TSO feature datasets generated on-the-fly during model training (weather FE config is an Optuna hyperparameter).

---

### Verification

1. `make lint` + `make test` — all new and existing tests pass
2. `make features` with real data → `processed/features_slim.parquet`
3. **EP comparison:** compare `features_slim.parquet` columns and values against EP's v5_slim output. Tolerance: RMSE < 0.01 per column
4. Leakage validation passes for all default feature lists; all known-bad features caught
5. Weather FE test: compare `WeatherWindPowerFE` on real offshore data against EMA output
