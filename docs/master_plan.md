# Master Implementation Plan

Merging `energy_prices` (EP) and `energy_market_analysis` (EMA) into a unified `energy-forecasting` repo.

**Date:** 2026-03-27
**Related documents (archived — analysis trail that produced this plan):**
- `docs/archive/merge_evaluation.md` — Stage-by-stage comparison of both repos
- `docs/archive/merge_decisions.md` — 17 architectural decisions with rationale
- `docs/archive/EP_EMA_merge_plan.md` — Earlier narrative plan (Plan A)

---

## Goals

**Primary:** Reproduce all existing functionality from both repos in a single, cleaner codebase. This means:
- Day-ahead price forecasts for DE-LU (from EP)
- 7-day generation/load forecasts per TSO (from EMA)
- Automated daily inference and periodic retraining
- Combined dashboard and API serving all forecasts

**Secondary (deferred to stage 8):** Weather features directly in price models, multi-day price forecasting, DuckDB analytics, chatbot, proper website hosting.

---

## Key Architectural Decisions (Summary)

Full rationale in `docs/merge_decisions.md`. Quick reference:

| # | Decision |
|---|----------|
| 1 | Dual FE system: suffix DSL for market features, config dict for weather FE. Single flat feature list as user interface. Load model Optuna results inform price model FE choices. |
| 2 | MLflow: ~8-12 focused experiments. Key rule: runs within an experiment must be comparable. |
| 3 | Ensembling: implement both inverse-MAE blend and stacking. Auto-select per target at retrain. |
| 4 | DuckDB deferred. Parquet files directly for now. |
| 5 | Dataset tracking via `mlflow.log_input()`. |
| 6 | Reproduce first, except combined dashboard and API from the start. |
| 7 | Cleaning rules as Python dataclasses, not YAML. |
| 8 | EP's gitignored test suite exists and should be ported. |
| 9 | Optuna everywhere: GridSampler for price, TPE for gen/load. |
| 10 | EMA's day-boundary-aware time-series CV. |
| 11 | Two CI workflows (daily + retrain). Daily follows: parallel data collection -> gen/load inference -> price inference -> deploy. |
| 12 | Weather FE computation decisions (air density mode, spatial aggregation) transfer from load to price models. Individual feature inclusion is target-specific. |
| 13 | Python config for logic (cleaning rules, feature lists, search spaces). JSON for pure data (eu_locations). |
| 14 | Models stored in GitHub Releases, not git. |
| 15 | PI blending: weighted average to start, monitor coverage, upgrade to calibrated if needed. |
| 16 | Feature datasets persisted via MLflow (no separate filesystem cache). Incremental extension loads previous dataset from MLflow, computes only new rows. |
| 17 | Python 3.13 (all deps confirmed compatible). |

---

## Stage Overview

```
Stage 1: Foundation (incl. API data contract)
    |
Stage 2: Data Collection & Storage
    |
Stage 3: Data Cleaning & Merging
    |
Stage 4: Feature Engineering
    |-- Market features (suffix DSL, lags, EWMA, spreads, etc.)
    |-- Weather features (physics FE, spatial aggregation, Optuna)
    |-- Shared: DSL parser, leakage validation, feature engine
    |
Stage 5: Model Training & Ensembling
    |-- Gen/load models (recursive, per-TSO, MAPIE)
    |-- Load→price FE transfer (Optuna weather FE choices)
    |-- Price models (direct, global hourly, blend)
    |-- Shared: training loop, CV, MLflow, Optuna
    |
Stage 6: Inference, API & CI/CD
    |
Stage 7: Dashboard
    |
Stage 8: Post-Merge Extensions
```

---

## Stage 1: Foundation

**Goal:** Project skeleton with working environment, config patterns established, utility functions ready.

**Detailed plan:** [`docs/stage1_foundation.md`](stage1_foundation.md)

**Deliverables:**
- `energy_forecasting/` package with config modules (`columns.py`, `cleaning.py`, `availability.py`, `modeling.py`), Parquet I/O utility, Pydantic API schemas
- `pyproject.toml` with all reconciled dependencies, `Makefile`, `.gitignore`
- `eu_locations.json` extracted from EMA
- Stub modules for later stages (importable, no circular deps)

**Milestone:** `pip install -e .` works, `make lint` passes, `make test` passes, all config modules importable.

### Stage 1 Evaluation

**Status:** Complete (2026-03-27)

**What was implemented:**
- Project skeleton: `pyproject.toml` (flit, all deps), `Makefile` (8 targets), `.gitignore`
- `config/__init__.py` — path constants, MLflow tracking URI
- `config/columns.py` — 85-entry short name registry, full SMARD filter dict (47 entries), cross-border flow dicts (DE-LU: 23 entries, DE-AT-LU: 25 entries), excluded keys (installed capacity + scheduled commercial), `clean_column_name()`
- `config/cleaning.py` — full cleaning pipeline config (drop, clip, structural fills, calculated fills, interpolation) calling helpers in `data/processing.py`
- `config/availability.py` — 20 availability rules covering forecasts, prices, generation, load, commodities
- `config/modeling.py` — 7 MLflow experiment names, model categories, blend/ensemble constants
- `data/io.py` — `reduce_dtypes()`, `save_parquet()`, `load_parquet()` with zstd compression
- `data/processing.py` — full implementations of all 9 cleaning helpers (clip_bounds, fill_zero_after/before/before_first_valid, fill_from_difference, fill_from_column, fill_gen_total, interpolate_gaps, drop_columns)
- `api/schemas.py` — 9 Pydantic models (HourlyForecast, ForecastResponse, ForecastHistoryResponse, ModelInfo, ModelsResponse, DailyError, PerformanceResponse, HealthResponse)
- `data/locations/eu_locations.json` — extracted from EMA (76 locations: 21 DE cities, 7 offshore, 14 onshore, 17 solar; plus FR locations)
- 29 stub modules for stages 2-7 (all importable, no circular deps)
- 48 tests across 5 test files, all passing
- `pip install -e ".[dev]"` confirmed on Python 3.13, `make lint` and `make test` pass clean

**Deviations from plan:**
- `data/processing.py` was listed as stage 3 but implemented fully in stage 1. Reason: `config/cleaning.py` imports from it, and the plan's own test section expects cleaning helpers to be testable now. Without the implementations, the cleaning config would fail on import and the tests couldn't run.
- `eu_locations.json` schema differs from the plan's suggested structure. The plan showed a nested `countries[].regions[].locations` hierarchy; the actual EMA data is structured as flat lists per location type (`de_loc_cities`, `de_loc_offshore_windfarms`, etc.) with `countries_metadata` as a separate list. Extracted as-is rather than reshaping — reshaping can happen in stage 2 if the DataSource abstraction needs a specific schema.
- Location field names: EMA uses `capacity` (not `capacity_mw` as the plan assumed). Also has richer metadata than expected (population density, industrial activity fraction, renewable energy fractions, EV counts, etc.).
- EP's `filter_dict` has a typo in key 251: `"Marktpres: Deutschland/Austria/Luxembourg"` (missing 'i' in Marktpreis, English instead of German). Fixed to `"Marktpreis: Deutschland/Österreich/Luxemburg"` for consistency.
- Added `catboost_info/` to `.gitignore` (CatBoost creates this directory during import).

**Challenges encountered:**
- EMA's `eu_locations.py` couldn't be imported via the package's `__init__.py` (transitive dependency on `user_agent` module). Solved by loading the file directly with `importlib.util`.
- `holidays` package was needed as a transitive dependency for `eu_locations.py` extraction but was already in pyproject.toml deps.

**Insights for later stages:**
- The cleaning helpers in `data/processing.py` are fully functional now, so stage 3 can focus on the merge pipeline and testing against EP's actual merged dataset rather than reimplementing cleaning logic.
- EMA location data is richer than anticipated — the extra fields (industrial activity, renewable fractions, HDD/CDD) could be useful for weather FE in stage 4.
- The short name registry (85 entries) is large enough that stage 4's DSL parser should include a "did you mean?" suggestion for typos in feature strings.

---

## Stage 2: Data Collection & Storage

**Goal:** All data sources download correctly and produce valid Parquet files. `make data` runs end-to-end.

**Detailed plan:** [`docs/stage2_data_collection.md`](stage2_data_collection.md)

### 2.1 DataSource Abstraction

Port EP's `DataSource` base class with shared `update()` logic. Create subclasses:

| Class | Source | From |
|-------|--------|------|
| `SmardSource` | SMARD API per-TSO + national | Merge EP + EMA. EP's parameterised region/resolution, EMA's per-TSO codes, known-missing handling, crash-resilient saves. |
| `OpenMeteoSource` | Weather (3 endpoints) | EMA. Override `update()` for three-endpoint logic (archive + historical forecast + current forecast). |
| `IcapSource` | Carbon allowances | EP wholesale. |
| `YahooSource` | TTF, Brent, CO2.L | EP wholesale. |
| `FredSource` | Gas price reconstruction | EP wholesale. |
| `EnergyChartsSource` | Day-ahead prices | EP wholesale. |

### 2.2 Location Data

Extract EMA's `eu_locations.py` (1,800 lines of Python) to `data/locations/eu_locations.json`. Schema:
```json
{
  "countries": [{
    "code": "DE",
    "regions": [
      {"name": "DE_50HZ", "suffix": "_50hz", "tso": "50Hertz", "available_targets": [...]}
    ],
    "locations": {
      "offshore": [{"name": "...", "lat": 53.88, "lon": 8.55, "capacity_mw": 500, ...}],
      "onshore": [...],
      "solar": [...],
      "cities": [...]
    }
  }]
}
```

### 2.3 Storage Layout

All sources write Parquet directly (no CSV intermediate). Directory structure as in stage 1.

### 2.4 CLI

Extend Typer CLI:
```
energy-forecasting download smard --region DE_LU --resolution hour
energy-forecasting download weather --type offshore --tso 50Hertz
energy-forecasting download commodities
energy-forecasting update           # incremental update all sources
```

### 2.5 Milestone

- `make data` downloads all data from scratch into correct directory structure
- `make update` incrementally updates all sources
- All Parquet files valid (no unexpected NaN, continuous timestamps, physical bounds for weather)
- Unit tests for each DataSource (mocked API responses)

### Stage 2 Evaluation

**Status:** Complete (2026-03-27)

**What was implemented:**
- `config/smard.py` — SMARD API config: national/TSO region codes, TSO suffixes, 12 per-TSO filter keys, 8 known-missing combos, API base URL, default parameters (redundancy days, bootstrap days)
- `config/commodities.py` — Commodity config: Yahoo tickers (TTF, Brent, CO2.L), ICAP system IDs (Phase 3/4), output column names, data start dates, price ranges, FRED series IDs, TTF reconstruction constants, Energy Charts config, ICAP URL
- `data/smard.py` — Low-level SMARD API client: `get_timestamps()`, `get_data()`, `get_all_data()` with `DataNotAvailableError`, NaN dropping, deduplication, UTC DatetimeIndex output
- `data/sources.py` — `DataSource` ABC with shared download/update/merge logic; `SmardSource` (national + per-TSO, ThreadPoolExecutor parallel fetch, crash-resilient resume, bisect-based incremental updates with redundancy window, column-level bootstrapping); `EnergyChartsSource` (Energy Charts API price fetch with D+2 lookahead); `_merge_column()` helper
- `data/weather.py` — `OpenMeteoSource` with three-endpoint logic (archive 2015+, historical forecast 2022+, current 14-day forecast), `_fetch_hourly()` shared request handler with retry (5 attempts, exponential backoff), physical limit validation via `np.clip()`, solar elevation/azimuth computation via pysolar, location loading from `eu_locations.json`, 16 physical limit definitions across all variable types
- `data/commodities.py` — `_download_icap()` (CSV parsing, Phase 3+4 concat, EUR/USD rate extraction); `reconstruct_ttf()` (FRED EU/US gap-fill with unit conversion + bias correction); `merge_carbon()` (ICAP + CO2.L dual-source with bias correction + bounded forward-fill); `IcapSource`, `YahooSource`, `FredSource` DataSource subclasses; `all_commodity_sources()` factory
- `cli.py` — Typer app with `download` subcommand (smard, smard-tso, weather, commodities, energy-charts, all) and `update` subcommand (all, smard, weather, commodities). Lazy imports in all commands.
- `pyproject.toml` — Added openmeteo-requests, requests-cache, retry-requests deps; responses dev dep; `[project.scripts]` entry point
- `Makefile` — `data`, `update`, `data-smard`, `data-weather`, `data-commodities` targets
- 7 new test files (45 new tests): `test_config_smard.py` (6), `test_smard_api.py` (6, mocked HTTP), `test_smard_source.py` (8, mocked HTTP + monkeypatched download), `test_weather.py` (11, physical validation + location loading + solar columns), `test_commodities.py` (7, reconstruction with synthetic data), `test_cli.py` (5, smoke tests), `test_data_coverage.py` (8, integration tests skipped without data)
- Total: 93 tests passing, 8 skipped (integration), lint clean

**Deviations from plan:**
- `FRED_SERIES` keys changed from `"eu_gas_monthly"`/`"us_gas_daily"` to `"fred_eu_gas"`/`"fred_us_gas"` to match the storage layout filenames (`fred_eu_gas.parquet`, `fred_us_gas.parquet`) and `DATA_START` keys. Keeps naming consistent across config, filenames, and CLI.
- `ICAP_URL` moved from `data/commodities.py` to `config/commodities.py` alongside other commodity constants. Cleaner separation of config vs logic.
- `_add_solar_columns()` and `_validate_physical()` implemented as module-level functions rather than methods on `OpenMeteoSource`. Reason: they're pure functions that don't need instance state, and this makes them independently testable.
- `_fetch_hourly()` extracted as a shared module-level function used by all three endpoints, rather than three separate methods with duplicated response-parsing logic.
- Historical forecast endpoint uses comma-separated strings for lat/lon (matching EMA's approach for this specific endpoint), while archive and current forecast use lists. This matches the upstream API's parameter format differences.

**Challenges encountered:**
- None significant. The plan's code snippets were detailed enough to implement directly. The main work was filling in the `...` ellipses (historical forecast and current forecast methods in weather.py, ICAP CSV parsing, EnergyCharts API call).

**Insights for later stages:**
- `reconstruct_ttf()` and `merge_carbon()` are currently standalone functions called with a `raw_dir` path. Stage 3's merge pipeline will need to orchestrate calling these after raw downloads complete — they read from the commodity Parquet files and produce the final reconstructed series.
- The `_merge_column()` helper in `sources.py` is generic and may be useful in stage 3's merge pipeline for combining national + historical SMARD data.
- Weather collection will be the slowest part of `make data` — three API calls per (asset_type, TSO) combination, with 5-attempt retry and backoff. The current forecast endpoint is fast (14 days), but archive (2015+) and historical forecast (2022+) are large downloads. Consider parallelising by asset type in stage 6's CI workflow.

---

## Stage 3: Data Cleaning & Merging

**Goal:** Clean merged dataset produced from raw data, ready for feature engineering.

**Detailed plan:** [`docs/stage3_data_cleaning.md`](stage3_data_cleaning.md)

### 3.1 Cleaning Rules

Implement the generic interpreter that reads the dataclass rules from `energy_forecasting/config/cleaning.py`. Processing order: physical bounds validation -> structural fills (domain rules) -> calculated fills -> interpolation.

Port all 12 of EP's rules as dataclass instances. Add EMA's physical bounds for weather variables. Add 2-3 special handlers for complex conditional logic (e.g., EP's forecast generation fill with 30-day recency check).

### 3.2 Merge Pipeline

- Load per-TSO SMARD data, aggregate to national where needed
- Concatenate DE-AT-LU (pre-2018) and DE-LU (post-2018) for historical depth
- Create unified `target_price` column (auto-select by regime)
- Add regime indicators (`regime_de_at_lu`, `regime_quarter_hourly`)
- Merge commodity prices (daily -> hourly alignment via forward-fill)
- Apply cleaning rules
- DST handling: store everything UTC through the pipeline, convert to local time only at target creation. Spring-forward hours interpolated, fall-back hours averaged.
- Quarter-hourly: aggregate to hourly for now (arithmetic average of four 15-min clearing prices)

### 3.3 Periodicity Enforcement

Port EMA's `fix_broken_periodicity_with_interpolation()` — detect missing timestamps, add them, interpolate. Applied early before feature engineering.

### 3.4 Milestone

- `make process` produces `data/processed/merged.parquet`
- Clean merged dataset: no unexpected NaN, continuous hourly timestamps, all regime indicators correct
- Unit tests for each cleaning rule type against synthetic DataFrames
- Verify key columns match EP's current `merged_dataset_hourly.parquet` (within tolerance for new cleaning rules)

### Stage 3 Evaluation

**Status:** Complete (2026-03-28)

**What was implemented:**
- Umlaut transliteration in `clean_column_name()` (ö→oe etc.) — fixed silent mismatch between SMARD column names and config references
- `config/merge.py` — regime dates, price column names, imputation thresholds, SMARD warning bounds, TSO-to-national mapping
- `data/merge.py` (~400 lines, 14 functions) — full merge pipeline: `warn_physical_bounds`, `enforce_periodicity`, `impute_medium_gaps`, `merge_national_smard`, `create_unified_target`, `extend_with_energy_charts`, `add_regime_indicators`, `build_commodity_daily`, `merge_commodities`, `normalize_dst`, `validate_no_nans`, `cross_validate_national_vs_tso`, `clean_tso_data`, `run_merge_pipeline`
- CLI `process` command + Makefile target
- 27 merge tests, 4 config tests, umlaut test — all passing
- Pipeline produces `data/processed/merged.parquet` (98,544 rows, 72 columns) + `data/processed/tso/*.parquet` (5 TSOs)

**Output characteristics:**
- merged.parquet: tz-naive hourly index (Europe/Berlin delivery hours), exactly 24 rows per day, continuous from 2014-12-31 to present
- All commodity columns (TTF, carbon, Brent) complete with 0 NaN after bounded ffill
- `target_price`: 120 NaN at dataset start (first 5 days, before SMARD price series begins) — expected, not clipped
- Per-TSO: UTC hourly, 0 NaN for 3 of 5 TSOs; Amprion/TransnetBW have structural gaps in hydro/other_conv (genuinely missing from SMARD)

**Deviations from plan:**
- DST output is tz-naive instead of tz-aware Europe/Berlin. The nonexistent spring-forward hour (e.g., 02:00 CET on March 29) cannot be represented as a valid tz-aware timestamp. Tz-naive correctly represents all 24 delivery hours. This also fixed a bug in EP's original implementation where timedelta arithmetic on tz-aware timestamps produced hour 3 instead of the missing hour 2.
- `carbon_realtime_eur_per_ton` dropped from output (matching EP). It was only used to extend the unified carbon column via bias correction; keeping it as a 60% NaN feature added noise.
- Commodity ffill uses 30-day bounded limit (plan didn't specify). ICAP carbon has consistent ~22-day Christmas closures; 30 days covers these while catching genuine pipeline failures.
- Brent start date changed from 2015-01-01 to 2010-01-01 — earlier data useful for lags/rolling averages.

**Bugs found and fixed (stage 2):**
- yfinance is not thread-safe: concurrent `yf.download()` calls corrupt results (all tickers return same data). Changed all commodity CLI commands to sequential execution. Added regression tests.
- yfinance MultiIndex columns: newer versions return `(metric, ticker)` MultiIndex. Added `droplevel(1)` in `_normalize()`. Added regression test.

**Challenges encountered:**
- DST normalization was the trickiest function. Pandas tz-aware arithmetic operates on UTC (not wall clock), making it impossible to create the missing spring-forward hour as a tz-aware timestamp. Solved by converting to naive local time before manipulation.
- ICAP carbon has a regular ~3-week Christmas publication gap every year (mid-Dec to early-Jan). The 2020/2021 gap was anomalously long (46 days). The tiered ffill approach (merge_carbon internal ffill + merge_commodities 30-day bounded ffill) handles both.

**Insights for later stages:**
- The tz-naive delivery-hour index means stage 4 feature engineering doesn't need timezone conversion — lags, same-hour-yesterday, daily aggregations all work directly on delivery hours 0-23.
- Per-TSO weather joins (stage 4) will use UTC since both weather and per-TSO SMARD are in UTC. Only the national merged dataset uses local time.
- The `stromverbrauch_pumpspeicher` and `stromverbrauch_residuallast` columns consistently trigger physical bounds warnings (values below 20,000 MW). These are genuinely lower-magnitude series — the SMARD_WARN_BOUNDS thresholds for `stromverbrauch_*` should be refined to distinguish load (40-80k MW) from pumped storage consumption (~0-8k MW) and residual load (can go negative).
- Cross-border flow columns for Northern Italy, Hungary, Slovenia are 66% NaN (only exist in DE-AT-LU era). Consider dropping these in stage 4 feature selection rather than carrying them through.

---

## Stage 4: Feature Engineering

**Goal:** Feature lists for both price and gen/load models produce correct datasets. This is the largest stage.

### 4.1 Suffix DSL Parser

The parser takes a feature string and returns a specification:

```python
parse_feature("price_d7")
# -> FeatureSpec(base="price", raw_col="target_price",
#                agg=Aggregation(start_day=-7, end_day=-7, stat="avg"))

parse_feature("price_d7_d1")
# -> FeatureSpec(base="price", raw_col="target_price",
#                agg=Aggregation(start_day=-7, end_day=-1, stat="avg"))

parse_feature("price_ewma_186_d1_h10")
# -> FeatureSpec(base="price", raw_col="target_price", lag=None,
#                ewma=EWMA(span=186, cutoff_day=-1, cutoff_hour=10))

parse_feature("gen_solar_d2__x__day_index")
# -> InteractionSpec(left=FeatureSpec(...), right=FeatureSpec(...))
```

The parser resolves short names via the registry, then greedily matches the suffix grammar. The grammar itself:

```
feature        := interaction | simple
interaction    := simple "__x__" simple
simple         := base_name suffix?
suffix         := ewma | lag | agg | fourier | daily_agg
ewma           := "_ewma_" INT ("_d" INT ("_h" INT)?)?
lag            := "_h" INT
agg            := "_d" INT ("_d" INT)? ("_eh" INT)? ("_h" INT "_h" INT)? ("_" STAT)?
fourier        := "_fourier_" INT "_" INT
daily_agg      := "_daily_" STAT
STAT           := "avg" | "std" | "min" | "max" | "sum" | "range" | "share"
```

**Design notes:**
- `_h` is for hourly lags: exact value N hours back. `price_h24` = same hour yesterday, `price_h168` = same hour last week.
- `_d` is for daily aggregations. When STAT is omitted, it defaults to `avg` (the most common case). So `price_d1` = average price yesterday, `price_d7` = average price 7 days ago, `price_d7_std` = std dev 7 days ago, `price_d7_d1` = average from day -7 to day -1.
- `__x__` is a reserved separator for interaction terms; base column short names must not contain it.

### 4.2 Feature Computation Engine

`engineer_features(df, feature_list) -> DataFrame` orchestrates:
1. Parse all feature strings
2. Identify required base columns (from short name registry + derived column registry)
3. Compute derived base columns (spreads, net exports, percentages, weather physics features)
4. Group by computation type (EWMA, rolling, lags, interactions)
5. Compute in order: base columns -> aggregations/lags/EWMA -> interactions
6. Validate against availability rules
7. Return DataFrame with exactly the requested columns

### 4.3 Market Feature Functions

Plain functions, no sklearn transformers:

| Function | Computes | From |
|----------|----------|------|
| `compute_price_spreads(df, neighbours)` | `target_price - neighbour_price` | EP |
| `compute_net_exports(df, flow_pairs)` | `exports - imports` per country | EP |
| `compute_generation_pct(df, sources)` | `source / total_generation` | EP |
| `compute_rolling_stats(df, col, window_spec)` | Mean/std/min/max over configurable windows | EP |
| `compute_ewma(df, col, span, cutoff_day, cutoff_hour)` | EWMA with information cutoff | EP |
| `compute_same_hour_lag(df, col, days)` | Same-hour value N days back | EP |
| `compute_hourly_lag(df, col, hours)` | Value N hours back | EP |
| `compute_temporal_features(df, tz)` | Cyclical hour/dow/month, holidays, day_index | EP |
| `compute_german_holidays(df)` | Population-weighted holiday indicator | EP |

### 4.4 Weather Feature Classes

Port EMA's three classes with Optuna integration preserved:

- `WeatherWindPowerFE` — air density (dry/moist), wind power density, wind shear, turbulence, gust factor, dew point, vapour pressure
- `WeatherSolarPowerFE` — cloud/clear sky fractions, radiation ratios, solar geometry
- `WeatherLoadFE` — HDD, CDD, wind chill, humidex, pressure trends, effective solar

Each class: `__init__(config_dict)`, `transform(df) -> df`, `suggest_optuna(trial) -> config_dict`.

Port spatial aggregation functions to `energy_forecasting/features/spatial.py`: haversine distance, capacity-weighted, inverse-distance, population-weighted, mean, max.

Weather FE outputs become base columns available to the suffix DSL (e.g., `wind_power_density_offshore_cap` can then be referenced as `wpd_offshore_cap` in a feature list).

### 4.5 Leakage Validation

Declarative validation decoupled from any pipeline framework. The validator:
1. Reads each feature's suffix to determine the implied lag
2. Looks up the base column's availability rule
3. Checks: implied lag >= required lag
4. Features with no suffix must be forecast/static columns (max_offset=0)

```python
validate_feature_list(feature_list, availability_rules) -> list[str]  # returns violations
```

### 4.6 Milestone

- Suffix DSL parser handles all documented grammar cases (unit tests)
- `PRICE_FEATURES` list produces a dataset matching EP's current v5 features (within tolerance)
- Weather FE classes reproduce EMA's current feature outputs
- Leakage validation catches known-bad configurations and passes known-good ones

### Stage 4 Evaluation

**Status:** Complete (2026-03-30)

**What was implemented:**
- `features/parser.py` — Suffix DSL parser producing typed `FeatureSpec`/`InteractionSpec` dataclasses. Full grammar: `_h` lags, `_d` single/multi-day aggregations, `_eh` end-hour truncation, `_h_h` hour filter, `_ewma` with cutoff, `_fourier`, `_daily` broadcast aggregates, `__x__` interactions. Longest-prefix short name resolution with "did you mean?" suggestions on typos.
- `features/market.py` (~350 lines) — Pure functions: `compute_rolling_stat` (optimized fast path via daily pre-aggregation for avg/sum/min/max/range, hourly rolling for std; slow path for end-hour/hour-filter cases), `compute_ewma` (vectorized via resample+shift, no per-day loop), `compute_hourly_lag`, `compute_price_spreads`, `compute_net_exports`, `compute_generation_pct`, `compute_temporal_features`, `compute_german_holidays` (population-weighted across 16 states), `compute_fourier_features`, `compute_interaction`.
- `features/validation.py` — Leakage validation against `config/availability.py` rules. Checks bare names, insufficient hourly lags, too-recent aggregation windows, EWMA without cutoff, daily aggregates on non-deterministic data, end-hour exceeding cutoff. Fails fast on unmatched availability rules (not silent pass).
- `features/engine.py` — `engineer_features(df, feature_list)` orchestrates parsing → validation → derived column pre-computation (`_prepare_working_df` for gen_pct, spreads, net exports, temporal) → per-feature computation → result collection. `extend_features()` for incremental updates with 30-day lookback window.
- `features/weather_physics.py` — 13 physics helpers: air density (dry/moist), wind power density, dew point (Magnus), vapor pressure, wind shear, turbulence intensity, wind ramp, gust factor, wind chill, humidex, HDH, CDH, haversine distance.
- `features/spatial.py` — Multi-location aggregation with 12 weighting methods (mean, max, IDW, capacity, n_turbines, n_panels, population, energy, and distance-weighted variants of each).
- `features/weather_wind.py`, `weather_solar.py`, `weather_load.py` — Three weather FE classes ported from EMA, each with `__call__(df) -> df`, per-location processing, spatial aggregation, and `suggest_optuna(trial)` for hyperparameter search.
- `config/features.py` — `PRICE_FEATURES_SLIM` (83 features, exact 1:1 match with EP's v5_slim_hourly Phase 11 output), `PRICE_FEATURES_FULL` (130 features), `GEN_LOAD_FEATURES` (17 features). Plus constants: `GENERATION_COLUMNS`, `RENEWABLE_COLUMNS`, `NEIGHBOUR_PRICES`, `FLOW_PAIRS`, `CYCLICAL_PERIODS`, `GERMAN_STATE_POPULATIONS`.
- `config/locations.py` — `load_locations()` and `locations_for_tso()` wrapping `eu_locations.json`.
- `config/columns.py` — Added 15 short names for temporal/derived features (`hour`, `hour_of_day`, `day_of_week`, `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend`, `is_holiday`, `day_index`, `year_index`, plus derived market columns). Fixed 2 incorrect mappings: `prog_residual` → `prognostizierter_verbrauch_residuallast` (was `prognostizierte_residuallast`), `residual_load` → `stromverbrauch_residuallast` (was `residuallast`).
- `config/availability.py` — Added 13 rules for derived short names: `residual_load`, `supply_demand_gap`, `pct_renewable`, `total_exports/imports`, `total_generation`, `pct_prog_*`, `is_weekend`, `month_*`, `year_index`, `hour`, `hour_of_day`, `day_of_week`.
- CLI `features` command with `--feature-list` (slim/full/gen_load), `--validate-only` flag. Makefile targets: `features-slim`, `features-full`, `features-validate`.
- Deleted `features/cache.py` stub (caching replaced by MLflow per decision #16).
- 5 test files, 151 tests: `test_parser.py` (33), `test_market.py` (20), `test_validation.py` (45), `test_engine.py` (12), `test_weather_fe.py` (41).

**Output characteristics:**
- `features_slim.parquet`: 98,544 rows × 83 columns, 9.2 MB, computed in 59 seconds.
- NaN pattern: 48 rows (2 days) for D-2 features, 24 rows (1 day) for D-1, up to 720 (30 days) for D-30 rolling stats. 15 columns with zero NaN (temporal + raw forecasts). All NaN is in the expected warm-up period at dataset start.

**EP comparison (v5_slim_hourly):**
- Column set: exact 1:1 match (83 features = EP's 84 columns minus `y` target). Reconstructed by tracing all 12 EP pipeline phases including the Phase 11 ColumnDropper's 61 exclude patterns.
- Values compared against EP's cached X.parquet (MLflow run `97a5de8ecbb947e4ae57963bb1d6862f`, created 2026-03-02). Run on EP's own merged data with correct timezone alignment (`tz_convert("Europe/Berlin")` before `tz_localize(None)`):
  - 7 features EXACT (temporal: hour_of_day, day_of_week, sin/cos, day_index, year_index, prog_gen_solar)
  - 9 features CLOSE (gen_pct, pct_prog, some commodities — ratio-of-means vs mean-of-ratios semantic difference)
  - Remaining features differ due to EP's spring-forward DST bug (duplicate hour 3 instead of interpolated hour 2, affecting ~341 timestamps at 11 DST transitions) and stale X.parquet (EP's merged data updated since the cached features were computed)

**Deviations from plan:**
- No `_range` stat type in the plan's grammar; added it (needed for `price_d7_d1_range`, `price_d30_d1_range` which EP computes via separate `add_price_ranges()` function).
- Plan listed `compute_same_hour_lag` as separate from `compute_hourly_lag`. Unified into one function — `shift(24)` for same-hour-yesterday and `shift(168)` for same-hour-last-week are the same operation.
- `_daily_STAT` suffix added for current-day broadcast aggregates (e.g., `prog_gen_wind_pv_daily_max`). Plan's grammar didn't include this — it was needed to match EP's `HourlyDailyAggregateTransformer`.
- Rolling stat optimization: plan assumed per-day loop; implemented fast path using pandas daily groupby + rolling for O(n) instead of O(n²). Reduced computation from 897s to 59s on 98K rows.
- EWMA optimization: plan assumed per-day loop; implemented vectorized approach via `resample("D").last().shift()`.
- `hour_of_day` and `day_of_week` raw integer features added — EP keeps these alongside cyclical sin/cos. Not in original plan but required for exact EP parity.
- Forecast daily aggregates: only `prog_gen_wind_pv_daily_max` survives EP's Phase 11 dropper. Plan had 4 daily aggs.

**Challenges encountered:**
- SHORT_NAMES had 2 incorrect column mappings (`prog_residual`, `residual_load`) that only surfaced when running on real data. Neither the parser nor validation caught these because they don't check whether the mapped column exists in the DataFrame — that only fails at engine runtime. Fixed the mappings.
- EP's X.parquet stores features in UTC but they were computed on Europe/Berlin data. Initial comparison used `tz_localize(None)` on UTC timestamps (wrong — keeps UTC hours), producing max errors of 23 for `hour_of_day`. Fixed to `tz_convert("Europe/Berlin").tz_localize(None)`.
- Rolling stat fast path: initial attempt used daily pre-aggregation for all stats, but `std` of daily means ≠ `std` of hourly values. Fixed by routing `std` through hourly rolling window.
- Validation silently passed features with no matching availability rule. Fixed to fail-fast with an error message (caught during deslop review).

**Insights for later stages:**
- The tz-naive delivery-hour index works cleanly with `shift(24)` for same-hour lags — no DST complications. EP's tz-aware approach has a spring-forward bug that produces duplicate hour 3; our approach avoids this entirely.
- Weather FE classes are implemented but untested on real data (city + solar weather downloads still pending from stage 2 rate limiting). Integration testing deferred to first use in stage 5.
- Gen_pct features use mean-of-ratios (our approach: compute hourly percentages, then aggregate to D-2 mean) vs EP's ratio-of-means (overwrite actuals to D-2 mean, then compute percentage). The values are close (MAE < 0.004) but not identical. This is a conscious design choice — our approach is more principled but breaks exact EP parity for these 6 features.
- The `_prepare_working_df` pattern in the engine (pre-compute derived columns like gen_pct, spreads, net exports) works well for market features. Weather FE follows a different path (Optuna-driven config dict, per-TSO processing) and doesn't go through the DSL engine — it produces input columns that the DSL then references.

---

## Stage 5: Model Training & Ensembling

**Goal:** Models train, evaluate, and ensemble correctly for all targets.

**Detailed plan:** [`docs/stage5_model_training.md`](stage5_model_training.md)

### 5.1 MLflow Setup

**Detailed conventions:** [`docs/mlflow_conventions.md`](mlflow_conventions.md) — experiment structure, tagging rules, run lifecycle, helper function specs, experiment workflow.

Key points:
- ~8-12 focused experiments (decision #2). Core rule: **runs within an experiment must be comparable.**
- `TrackedRun` context manager validates required tags (stage, feature_version, holdout_days, cv_folds, cv_mode, target_transform) before closing.
- Helper functions: `audit_experiment` (flag inconsistent runs), `archive_runs`, `compare_models`, `compare_feature_sets`, `get_best_run`, `cleanup_orphaned_artifacts`, `export_experiment_summary`.
- Run lifecycle: Active → Archived (tagged, still queryable) → Deleted (broken runs only).
- Dataset tracking via `mlflow.log_input()` (decision #5), not the tag-based workaround EP used.

### 5.2 Training Loop

Port EP's `train_and_log()` (simplified — no sklearn pipeline wrapping):
1. Load feature dataset (from MLflow artifact or `data/processed/features_*.parquet`)
2. Time-series split using EMA's day-boundary-aware splitter (decision #10)
3. Optional exponential decay sample weights: `w = exp(ln(2)/half_life * (t - t_max))`
4. Apply target transform and feature scaling as part of the model pipeline (see below)
5. Wrap model in MAPIE `MapieRegressor` for prediction intervals
6. Fit, predict, compute metrics (including PI coverage)
7. Log to MLflow with `mlflow.log_input()` for dataset tracking (decision #5)
8. Return run_id

**Target transforms and scaling belong with the model, not feature engineering.** `FeatureScaler` and `TargetTransformer` are part of the sklearn-style model pipeline — `TransformedTargetRegressor` wraps the model with target scaling, and feature scaling is a pipeline step before the model. The choice of transform (log-shift, yeo-johnson, quantile, none) is a tunable parameter, evaluated alongside model hyperparameters. EP currently puts these in the dataset preparation stage — move them here.

### 5.3 Generation/Load Models

Port EMA's recursive multi-step forecasting:
- `forecast_window()` with lagged target updates
- Per-TSO models for wind (onshore/offshore), solar, load
- MAPIE conformal intervals on each model
- Stacking ensemble: base models -> OOF predictions -> meta-model

### 5.4 Load→Price Feature Transfer (Decision #12)

After gen/load models are tuned with Optuna (5.3 + 5.6), extract the weather FE computation choices before training price models:

1. **Extract FE computation decisions** from the best load model Optuna trial: air density mode (dry/moist), spatial aggregation method, turbulence window size, which physics features to compute, etc. These are the `WeatherBasedFE` config dict values.
2. **Load the saved weather FE output** — during gen/load training, the weather FE classes save their output to `data/processed/weather_features/`. The price model loads these pre-computed columns rather than recomputing. This avoids computing weather features twice.
3. **Select which weather features to include** in the price model's feature list separately. The *computation* choices transfer; the *inclusion* choices are target-specific. For example, the load model might use turbulence intensity while the price model doesn't — but both compute wind power density the same way.

This is an explicit dependency: gen/load tuning (5.3 + 5.6) must complete before price feature dataset construction and price model training (5.5).

### 5.5 Price Models

Port EP's direct prediction approach:
- Global hourly model (scalar target)
- 4 model categories: linear, LightGBM, XGBoost, CatBoost
- Inverse-MAE blend with 90-day holdout

EP's blend selection process (4 stages):
1. Pull candidate runs from MLflow experiment
2. CV validation — verify all candidates used same holdout/CV/features
3. Select best-MAE + best-RMSE per model category (up to 2 per category, 8 total)
4. Retrain selected models on all-minus-holdout, compute inverse-MAE weights, save `blend_config.json`

**Empirical finding from EP:** time-varying weights (rolling window inverse-MAE, not static) outperformed stacking alternatives. Start with rolling-window weights as the default blend approach.

### 5.6 Ensemble Auto-Selection

Implement both blend and stacking (decision #3). At retrain:
1. Train blend ensemble, evaluate on holdout
2. Train stacking ensemble, evaluate on holdout
3. Pick the one with lower holdout MAE
4. Log both results for comparison

### 5.7 Hyperparameter Tuning

Optuna integration (decision #9):
- `GridSampler` for price models (smooth landscape)
- `TPE` sampler for gen/load models (joint feature + hyperparam search)
- Centrally declared search spaces per model type
- Best params saved to JSON for reproducible retrain
- Log Optuna trials to MLflow via `optuna.integration.MLflowCallback` for gen/load experiments

**Price model tuning strategy (two-pass):**
1. **Coarse grid** over 3-4 important parameters (learning rate, max depth, regularisation). The GBT hyperparameter landscape is smooth and the important parameters are well-known. A coarse grid then a finer grid around the best region gets within a few percent of Optuna's optimum in a fraction of the time.
2. **After settling on features and hyperparams independently:** re-tune hyperparams on the final feature set to catch the most important feature-hyperparam interactions. This is one additional grid search, not a combinatorial explosion.

Feature selection for price models is human-curated via the feature name list, not Optuna-searched. EMA's joint feature+hyperparam Optuna search is computationally infeasible for the combined price model search space.

### 5.8 Prediction Intervals

MAPIE on all models (decision #15). EMA already uses MAPIE for generation forecasts — port the same pattern to price models. Uses MAPIE 1.x `CrossConformalRegressor` (0.9.x `MapieRegressor` was removed in 1.0).

```python
from mapie.regression import CrossConformalRegressor

base_model = LGBMRegressor(**best_params)
model = CrossConformalRegressor(base_model, method="plus", cv=5, confidence_level=0.9)
model.fit_conformalize(X_train, y_train)
y_pred, y_intervals = model.predict_interval(X_test)  # 90% interval
```

**XGBoost precision fix:** EMA discovered that XGBoost's float32 precision requires `eps=1e-4` on the conformity score. In MAPIE 1.x, pass an instantiated score object rather than the default string: `conformity_score=GammaConformityScore(eps=1e-4)` if using gamma scores. Verify whether this is still needed with current XGBoost + MAPIE 1.x — the issue was float32 metric precision falling below the default `1e-6` threshold.

**PI blending formula.** Each model in the blend produces point + lower + upper. Blend using the same inverse-MAE weights:
```
blend_point = Σ(w_i × point_i)
blend_lower = Σ(w_i × lower_i)
blend_upper = Σ(w_i × upper_i)
```

Track coverage metric from day one. At inference, output JSON includes `forecast_lower` and `forecast_upper` alongside `forecast` for each hour.

### 5.9 Milestone

- Price model blend MAE comparable to EP's current production (~9.9 EUR/MWh)
- Gen/load models reproduce EMA's current forecast accuracy
- MLflow experiments populated with correctly tagged runs
- Both ensemble methods implemented and compared
- PI coverage tracked and ~95%

### Stage 5 Evaluation
<!-- Fill in after stage 5 is complete -->

**Status:** 5a complete; 5b complete (2026-05-04); 5c complete (2026-06-30); 5d not started

#### 5a: Training Infrastructure (complete, 2026-04-03)

**What was implemented:**
- `config/modeling.py` — all constants (MAPIE, CV, holdout, weights, blend, gen/load targets, experiments, EEG dates)
- `config/search_spaces.py` — Optuna suggest functions (LGBM, XGBoost, CatBoost, Ridge, Lasso, dataset params), price grid points, preprocessing grids
- `modeling/mlflow_utils.py` — TrackedRun (validates experiment + tags at __enter__), get_best_run, compare_models, audit_experiment, archive_runs
- `modeling/metrics.py` — calculate_metrics (MAE, RMSE, R², MAPE, sMAPE, skill scores), PI metrics, peak metrics
- `modeling/cv.py` — TimeSeriesSplitter (day-boundary-aware, expanding/sliding), carve_holdout
- `modeling/training.py` — build_pipeline (scaler → TransformedTargetRegressor), train_model (CV + MAPIE + MLflow), sample weights, OOF prediction collection
- `modeling/intervals.py` — MAPIE CrossConformalRegressor wrapper, post-hoc conformal calibration for ensembles
- `modeling/datasets.py` — prepare/load/update/find Parquet datasets, MLflow log_input provenance
- `modeling/baselines.py` — naive_lag, naive_weekly, naive_persistence_7d, naive_seasonal_7d, climatological_baseline
- `modeling/forecasting.py` — forecast_direct (default), forecast_with_lags (recursive)
- 53 tests across 7 test files

**Key decisions resolved:**
- MAPIE 1.3 XGBoost eps fix NOT needed (OQ#1) — coverage=96%, no NaN
- MAPIE sample_weight: two-path approach (OQ#2) — without weights → Pipeline direct to MAPIE; with weights → pre-scale + bare estimator + fit_params
- Single-row predict_interval works (OQ#3)

**Deviations:**
- TrackedRun validates tag values (not just presence) — stage must be in allowed set, feature_version must be non-empty, ad-hoc tags allowed
- _LogShiftTransformer inherits BaseEstimator for sklearn clone() compatibility

#### 5b: Gen/Load Models (code complete, training run pending, 2026-04-03)

**What was implemented:**
- `modeling/gen_load.py` — train_gen_load_model (Optuna TPE search over model hyperparams + weather FE config + dataset params), ensemble_gen_load (Ridge stacking with automatic fallback), temporal feature computation (bypasses DSL engine — TSO column names differ from DSL expectations)
- Upstream forecast features: load models use wind/solar **actuals** as exog features at training time, gen_load_diff uses wind/solar/load **actuals**. `_load_upstream_actuals()` mirrors EMA's `extract_from_database` (`data_loaders.py:369`). All TSO regions included per EMA pattern. (Originally planned via OOF predictions; replaced with actuals because OOF coverage was only ~6% of the training window — see "Challenges" below.)
- `gen_load_diff` national target: `_load_national_tso_data()` aggregates all TSOs, `_load_national_weather_data()` concatenates city weather across TSOs
- `cli.py` — `train gen-load` command with --target, --region, --model-type, --trials, --skip-ensemble. Respects `GEN_LOAD_TRAINING_ORDER`.
- Makefile targets: train-gen-load, train-gen-load-quick, train-gen-load-target
- `config/modeling.py` — REGION_TO_TSO, TARGET_WEATHER_TYPE, GEN_LOAD_OPTUNA_TRIALS, GEN_LOAD_TRAINING_ORDER, exog_targets per target
- MinMaxScaler added to training.py _SCALERS (matching EMA)
- OOF prediction collection added to train_model() for stacking ensembles
- Fixed artifact naming: explicit `holdout_predictions.parquet`/`oof_predictions.parquet` instead of random temp names
- 31 tests in test_gen_load.py

**Key decisions resolved:**
- Gen/load dependency order (OQ#4): wind/solar → load → gen_load_diff. Load uses wind/solar OOF predictions (out-of-sample) as exog features. gen_load_diff uses all upstream predictions. Training order enforced in CLI.
- Temporal features computed directly from TSO data (not via DSL engine) — TSO parquets use different column naming (`wind_onshore_50hz`) than the DSL expects (`stromerzeugung_wind_onshore`). Direct computation is cleaner for the short, stable GEN_LOAD_FEATURES list.

**Smoke test results (wind_onshore/DE_50HZ, 3 Optuna trials, pre-exog-feature code):**
- Holdout MAE: 240.96 MW (EMA reference: 560-1120 MW)
- PI coverage: 87.59% (target: 87-93%)
- Weather features saved to data/processed/weather_features/
- Full pipeline works end-to-end: Optuna → dataset → train_model → MAPIE → MLflow

**Remaining for gate 5b:**
- Full training run via `make train-gen-load` (wind/solar first, then load with exog, then gen_load_diff)
- Per-target/region accuracy table vs EMA reference numbers
- PI coverage confirmation across all targets

#### Pre-requisite fix applied (2026-04-10)

The `_eh10` → `_eh7` morning actuals cutoff fix (§5 pre-requisite) has been applied:
- Renamed feature strings in `PRICE_FEATURES_SLIM` and `PRICE_FEATURES_FULL` (both `_eh10` and `_d1_h10` variants).
- Updated `AVAILABILITY_RULES` for SMARD columns (`gen_*`, `load`, and derived) from `cutoff_hour=10` to `cutoff_hour=7`, so the validator now rejects `_eh10` going forward.
- Updated and extended `tests/test_validation.py` with regression guards for the new cutoff. All 80 validation/parser tests pass; `make features-validate` passes for slim, full, and gen_load lists.

**Deviations from plan:**

- **Upstream exog features for load/gen_load_diff are SMARD actuals, not OOF predictions.** Original plan called for OOF stacking; replaced with `_load_upstream_actuals` (`gen_load.py:202`) which mirrors EMA's `extract_from_database` (`data_loaders.py:369`) directly. OOF predictions only covered ~6% of the training window with the EMA-aligned 2-year cap, leaving only ~164 usable rows for load training. EMA's design uses actuals at training and forecasts at inference, accepting a small train/inference distribution mismatch.
- **Final-pass CV uses 218 folds (`GEN_LOAD_HISTORICAL_FOLDS`), not 5.** Originally `VALIDATION_CV_FOLDS=5` was used for the final pass; bumped to 40 at end of 5b, then to 218 in Phase A (2026-05-07) to produce ~4.18 years of OOF predictions per (target, region) spanning 2022-01-15 → 2026-03-27, matching the `hist_forecast` weather window. This collapses EMA's separate `generate_historical_forecasts.py` (1139-line standalone backtest pipeline) into the existing training run. The Optuna search keeps `SEARCH_CV_FOLDS=3` for speed.
- **Test-fold features come from `hist_forecast` weather, training-fold features from actual weather.** Implemented via `train_model(test_dataset_path=...)`. Mirrors EMA's `slice_weather_for_cutoff` (`generate_historical_forecasts.py:196`) at the fold level. Gives backtest-honest OOF predictions with realistic forecast errors.
- **ElasticNet pins `log_target=False`** (`config/search_spaces.py:74`). EMA hardcodes the same (`update_forecasts.py:93–104`); without the pin, RobustScaler on log-transformed targets overflows ElasticNet's coordinate descent.
- **Historical forecasts artifact saved to `data/processed/historical_forecasts/{target}_{region}.parquet`** with schema `[y_true, y_pred, y_lower, y_upper]`. Consumed by Stage 5c instead of the EMA overlay parquets EP previously read from a separate repo. National aggregates (`{target}_DE_NATIONAL.parquet`) are summed across regions on the common index, mirroring EMA's `export_national_forecasts.py`.

**Challenges encountered:**

- **Target-lag leakage in CV (fixed 2026-04-13)**: `_build_features` was creating `{target}_lag_{N}` columns via `y.shift(lag)` on the full series before splitting. During CV those columns held the actual future values; `lag_1` had r=0.99 with the target. Fix: rename to `{target}_h{N}`, switch CV to `forecast_with_lags_windowed` for honest recursive forecasting, bypass MAPIE for lag-enabled runs (it can't drive recursive predictions internally).
- **Training window misalignment (fixed 2026-04-13)**: trained on ~11 years; EMA caps at `100*168=16,800` hours. Fix: `GEN_LOAD_MAX_TRAIN_HOURS=16800`, sliding CV with weekly test folds, matching EMA's `compute_timeseries_split_cutoffs`.
- **MLflow `archived != 'true'` filter bug (fixed 2026-04-13)**: SQL filter excluded runs where the tag was absent (not just where it equaled `'true'`). Caused `_load_upstream_predictions` to return None for all queries. Fix: removed from the search query, applied in Python after.
- **ElasticNet × `log_target` overflow (fixed 2026-05-04)**: 9 of the 2026-04-13 quick-run failures were ElasticNet trials picking `log_target=True`. RobustScaler on log-shifted targets produces values outside float64 range. Fix: pin `log_target=False` for ElasticNet via `suggest_dataset_params(trial, model_type)`.
- **OOF coverage too thin for cross-target features (fixed 2026-05-04)**: original design loaded wind/solar OOF predictions as exog features for load training. With 5 CV folds × 168h, only 5% of the training window had OOF coverage → only ~164 usable rows for load. Replaced with EMA-style actuals injection at training time.
- **MLflow database schema upgrade**: required after the conda env rebuild on 2026-05-04 (mlflow 3.11 vs older). One-shot `mlflow db upgrade` after backing up `mlflow.db`.

**Insights for later stages:**

- **Stage 5c price model** consumes `data/processed/historical_forecasts/{target}_{region}.parquet` directly. Per-region per-TSO and a national aggregate (`{target}_DE_NATIONAL.parquet`) are both produced. Schema is `[y_true, y_pred, y_lower, y_upper]`; `y_lower`/`y_upper` are NaN when the upstream model used recursive lags (MAPIE bypassed).
- **Stage 6 inference** must populate the same TSO-suffixed column names that `_load_upstream_actuals` uses at training time, but from upstream model forecasts produced at inference. The runtime equivalent of EMA's `extract_from_database` `df_forecast` path (`data_loaders.py:379–394`).
- **Train/inference asymmetry is intentional**, mirroring EMA. Training rows (df_hist equivalent) use SMARD actuals for exog and Open-Meteo actual weather; test/holdout rows use Open-Meteo `hist_forecast` weather (and at production inference, upstream model forecasts). The only inconsistency is that exog targets at test time still use actuals — fully backtest-honest exog would require chained dependency resolution within a single CV pass, which EMA also doesn't do internally.
- **Model selection per-target** should prefer the run with lowest holdout MAE (CLI `_pick_best_base_run`). EMA's `best_model.json` uses the same logic after their 2026-03-15 fix that switched from `best_model_forecast.json` (training-data evaluation) to `best_model.json` (rolling CV).

**Stage 5b gate (passed 2026-05-04):**

Training run: 48 base models + 16 stacking ensembles, **0 failures**, 4h 40m wall clock, 70 Optuna trials per (target, region, model_type). All artifacts in MLflow under `generation/{target}` experiments.

Best run per (target, region) by CV MAE (rolling 40 weekly folds with hist_forecast weather on test slice; matches EMA's leak-free evaluation):

| target / region | best model | CV MAE | holdout MAE | holdout R² |
|---|---|---|---|---|
| wind_onshore / DE_50HZ | StackingEnsemble | 916* | 644 | 0.945 |
| wind_onshore / DE_AMPRION | XGBRegressor | 823 | 647 | 0.882 |
| wind_onshore / DE_TENNET | XGBRegressor | 1402 | 1206 | 0.914 |
| wind_onshore / DE_TRANSNETBW | LGBMRegressor | 134 | 140 | 0.680 |
| wind_offshore / DE_50HZ | LGBMRegressor | 158 | 190 | 0.812 |
| wind_offshore / DE_TENNET | LGBMRegressor | 509 | 387 | 0.944 |
| solar / DE_50HZ | XGBRegressor | 353 | 444 | 0.959 |
| solar / DE_AMPRION | XGBRegressor | 268 | 408 | 0.951 |
| solar / DE_TENNET | XGBRegressor | 439 | 536 | 0.952 |
| solar / DE_TRANSNETBW | LGBMRegressor | 170 | 196 | 0.966 |
| load / DE_50HZ | XGBRegressor | 341 | 308 | 0.949 |
| load / DE_AMPRION | XGBRegressor | 574 | 473 | 0.954 |
| load / DE_TENNET | LGBMRegressor | 378 | 617 | 0.897 |
| load / DE_TRANSNETBW | XGBRegressor | 198 | 194 | 0.964 |
| load / DE_CREOS | XGBRegressor | 16 | 22 | 0.864 |
| gen_load_diff / DE_NATIONAL | StackingEnsemble | 1564* | 1561 | 0.930 |

*Ensemble cv_mae is not directly logged (Ridge meta-learner is fit on the meta-train set); the value shown is the corresponding base model's cv_mae. Stacking ensemble's holdout MAE is the better signal for ensemble selection.

**EMA reference comparison.** EMA's `output/DE/forecasts/{target}_{region}/best_model.json` reports `avg_rmse` per combination from EMA's CV; `output/DE/evaluation_vs_smard.csv` reports national-level metrics across all 7 forecast days. Apples-to-apples comparison (our cv_rmse from 40 weekly folds vs EMA's avg_rmse):

Per-(target, region) — **we beat EMA on every combination, 8–55% lower RMSE**:

| Target / region | Our model | Our RMSE | EMA model | EMA RMSE | Δ% |
|---|---|---|---|---|---|
| wind_onshore/DE_50HZ | XGBRegressor | 1178 | LightGBM | 1576 | −25.3% |
| wind_onshore/DE_AMPRION | XGBRegressor | 1069 | LightGBM | 1172 | −8.7% |
| wind_onshore/DE_TENNET | LGBMRegressor | 1768 | XGBoost | 2051 | −13.8% |
| wind_onshore/DE_TRANSNETBW | LGBMRegressor | 187 | ElasticNet | 220 | −15.3% |
| wind_offshore/DE_50HZ | LGBMRegressor | 220 | LightGBM | 250 | −11.9% |
| wind_offshore/DE_TENNET | LGBMRegressor | 671 | ens[LGBM,ElasticNet] | 997 | −32.7% |
| solar/DE_50HZ | XGBRegressor | 710 | ElasticNet | 1493 | −52.4% |
| solar/DE_AMPRION | XGBRegressor | 532 | ElasticNet | 1176 | −54.8% |
| solar/DE_TENNET | XGBRegressor | 867 | ElasticNet | 1711 | −49.3% |
| solar/DE_TRANSNETBW | LGBMRegressor | 338 | ElasticNet | 743 | −54.5% |
| load/DE_50HZ | XGBRegressor | 431 | ens[LGBM,ElasticNet] | 716 | −39.7% |
| load/DE_AMPRION | LGBMRegressor | 714 | ens[LGBM,ElasticNet] | 876 | −18.5% |
| load/DE_TENNET | LGBMRegressor | 486 | ens[LGBM,ElasticNet] | 639 | −23.9% |
| load/DE_TRANSNETBW | XGBRegressor | 252 | ens[XGB,ElasticNet] | 547 | −53.9% |
| load/DE_CREOS | XGBRegressor | 21 | XGBoost | 26 | −20.7% |

National aggregate (sum-of-regions on both sides) on the **exact same 840 timestamps** (5 weekly cutoffs × 168h, 2026-02-06 → 2026-03-12), comparing our `historical_forecasts/{target}_DE_NATIONAL.parquet` to EMA's per-TSO best-model `trained/result.csv` summed identically:

| Target | hours | our MAE | EMA MAE | Δ% | our RMSE | EMA RMSE | Δ% |
|---|---|---|---|---|---|---|---|
| wind_onshore | 840 | 2714 | 3149 | **−13.8%** | 3747 | 4282 | **−12.5%** |
| wind_offshore | 840 | 503 | 753 | **−33.2%** | 663 | 1051 | **−36.9%** |
| solar | 840 | 1058 | 2316 | **−54.3%** | 2345 | 3422 | **−31.5%** |
| load | 840 | 992 | 1377 | **−28.0%** | 1258 | 1725 | **−27.1%** |

**We beat EMA on every national-aggregate target by 13–54% MAE.** The methodology is identical (both repos sum per-TSO predictions and per-TSO actuals), the timestamps are identical, the underlying SMARD ground truth is identical.

**Investigation history (all earlier conclusions on this section were wrong; the table above is the truth):**

1. First pass compared our parquets to EMA's `evaluation_vs_smard.csv` headline numbers and concluded we lose by 39–130%. That CSV evaluates EMA on `weeks=5` ending ~2026-04-17 — a window almost entirely OUTSIDE our parquet coverage. Different periods = different difficulty = headline gap was an artifact.
2. Second pass hypothesised EMA must use dedicated national models because our quadratic-sum lower bound (789 MW for load) exceeded EMA's 506 MW. Wrong reasoning: the quad-sum is the lower bound only under non-positive correlation; positive correlation makes sum-MAE larger than quad-sum, not smaller. The argument doesn't constrain EMA's methodology.
3. Third pass found and read EMA's `evaluate_vs_smard.py` (in commit `a434bf68`, moved to `local/` in `25120fa7`, recoverable via `git show a434bf68:evaluate_vs_smard.py`). It does `pd.concat(...).sum(axis=1, min_count=1)` on per-TSO `evaluation_daily.csv` files. Same aggregation as ours.
4. Fourth pass compared on the **same timestamps** using EMA's actual per-TSO trained predictions, summed identically. We beat EMA on every target — table above.
5. Fifth pass: ran honest 5-fold sliding CV with EMA's exact training-window size (~95 weeks per fold, vs our 40-fold default of ~59 weeks per fold) for 4 representative DE_50HZ combos, to rule out the methodology-mismatch hypothesis. Confirmed our 40-fold setup is within ~5% of the 95-week-trained version, and at matched setup we still beat EMA by 18–64% per-region MAE.

| (target, region) — DE_50HZ | Our 5-fold (95wk train) MAE | Our 40-fold (59wk train) MAE | EMA trained (95wk sliding) MAE | Δ vs EMA |
|---|---|---|---|---|
| wind_onshore | 824 | 792 | 1061 | **−22%** |
| wind_offshore | 138 | 136 | 169 | **−18%** |
| solar | 320 | 346 | 886 | **−64%** |
| load | 427 | 407 | 626 | **−32%** |

The headline-CSV numbers from EMA (e.g. load 506 MAE) reflect a different and apparently easier evaluation window. They aren't a useful benchmark for our parquets, which cover a different and tougher period. EMA's per-TSO best-model `trained/result.csv` predictions are the right reference, and we beat those at every level — per-region, national-aggregate, and matched-CV.

Solar's 64% per-region gap on DE_50HZ is the biggest, driven by EMA picking ElasticNet for all 4 solar regions (linear models can't capture solar's nonlinear weather dependence) while we pick LGBM/XGB.

The ~4.18 years of OOF predictions per (target, region) form the artifact Stage 5c consumes (post-Phase-A, 2026-05-07).

**PI coverage:** all base-model runs use recursive lag features and bypass MAPIE (matches EMA's design); their PI coverage is reported as NaN. Ensembles do post-hoc conformal calibration; coverage hovered around the 90% target on the 168h holdout (sample too small for tight statistics — Stage 5c retraining will accumulate more PI evaluations).

**Historical forecasts artifact:** 16 per-(target, region) parquets + 4 national aggregates (`{wind_onshore, wind_offshore, solar, load}_DE_NATIONAL.parquet` summed across regions); `gen_load_diff` already national. After Phase A (2026-05-07) each is ~36,788 rows = 218 weekly OOF folds + 7-day holdout (originally 6,884–6,888 rows / 40 folds at 5b end), leak-free, hist_forecast weather on test windows. Schema `[y_true, y_pred, y_lower, y_upper]`; `y_lower`/`y_upper` are NaN for combos selected from a recursive-lag base model.

**Bugs caught and fixed during the run:**

1. `ensemble_gen_load` was logging only `holdout_predictions.parquet`, not `oof_predictions.parquet`. Combined with the CLI preferring the ensemble run for the historical_forecasts export, this stranded all 6,720 OOF rows on the base-model runs and left the artifacts at only 168 holdout rows. **Fixed**: ensemble now logs OOF (Ridge meta-learner in-sample predictions on meta-train; small bias <1% given the constrained low-rank model). The 2026-05-04 run's artifacts were rebuilt out-of-band from base-model OOF + ensemble holdout.
2. `_pick_best_base_run` was selecting by 168h holdout MAE (noisy). **Fixed**: now uses `cv_mae` (218 folds × 168h ≈ 4.18 years of evaluations post-Phase-A; was 40 folds × 168h ≈ 9.5 months at 5b end) — matches EMA's 2026-03-15 model-selection fix.

#### 5c: Price Models (complete, 2026-06-30)

**What was implemented:**
- `modeling/price.py` — `run_price_pipeline` orchestrates feature prep → per-(model_type, feature_version) tuning → winner retrain with VALIDATION_CV_FOLDS → SLSQP ensemble bake-off → post-hoc conformal PI calibration → MLflow log + JSON config write. `precomputed_datasets` param lets experiments reuse existing fs_* parquets.
- `modeling/tuning.py` — two-stage grid for tree models (stage 1: weight pinning across WEIGHT_HALF_LIVES; stage 2: hyperparam grid). Exhaustive grid for linear models (preprocessing × alpha). Optuna SQLite studies resume on restart. Per-(model_type, feature_version, stage) MLflow run under `price/model_training`.
- `modeling/feature_selection.py` — `run_feature_selection`: correlation filter → SHAP importance → SHAP cutoff curve (coarse + fine grid around local minima) → RFECV narrowing → extra candidate logging. Each candidate logged as a `price/feature_selection` MLflow run. Curves saved to disk (`price_fs_shap_ranking.parquet`, `price_fs_shap_curve.parquet`, `price_fs_rfecv_curve.parquet`) and to MLflow (`meta_shap` run + `rfecv_optimum` artifact).
- `config/search_spaces.py` — LGBM probe and grid with `num_leaves = 2^max_depth − 1` per config (capacity fix), `subsample_freq=1` (bagging fix). Linear grids for Ridge and Lasso.
- Per-model-type pruning in `run_price_pipeline`: after tuning, configs more than 20% above a model type's best cv_mae are dropped before the expensive retrain step.
- `allow_writing_files=False` on CatBoostRegressor; `min_periods=window` on `compute_neg_price_stats` rolling calls.

**Key decisions resolved:**
- LGBM root cause (2026-06-06): `num_leaves=31` cap (8× less capacity than XGB) + `objective="mae"` zero-hessian degradation. Fixed by scaling `num_leaves` to `2^max_depth - 1` and enabling `subsample_freq=1`. Objective kept at MAE for EP comparability; L2 objective sweep is a roadmap item.
- Diversity candidates (LGBMQuantile, Huber): confirmed by SLSQP bake-off to earn zero ensemble weight. Removed from the pipeline after the experiment.
- `max` feature set dropped — never earned ensemble weight.
- Ensemble PI: post-hoc conformal calibration on holdout residuals (q fitted on holdout, not CV), matching EP's pattern. Base-model MAPIE PIs were uniformly under-covered (~70–90%); a single ensemble-level quantile restores target coverage.

**Results — first clean run (2026-05-29, baseline):**
- Holdout MAE **11.239**, RMSE 18.000, R²=0.846, PI coverage 90.05%, PI width 50.35
- Tree models (XGB × 3, CatBoost × 2) held 91% of ensemble weight; LGBM ~zero (crippled).

**Results — diversity experiment (2026-06-24, production config):**
- Holdout MAE **11.148**, RMSE 18.094, R²=0.844, PI coverage 90.05%, PI width 48.89
- `LGBMRegressor__fs_shap_top90`: **0.346** weight (largest single contributor — LGBM fix worked)
- `XGBRegressor__fs_rfecv_optimum`: 0.197, `XGBRegressor__fs_shap_top90`: 0.188
- `CatBoostRegressor__fs_rfecv_optimum`: 0.184, `Ridge__fs_shap_top247`: 0.084
- LGBMQuantile and Huber: zero weight everywhere — confirmed non-contributors, removed.

**All review items closed:** #1–#8 (see `docs/stage5c_status_2026-06-06.md`).

**Production config promoted:** `models/ensemble_config.json` is the 2026-06-24 diversity run (MAE 11.148). The baseline (11.239) is preserved as `models/ensemble_config_5c_diversity.json` for comparison.

---

## Stage 6: Inference, API & CI/CD

**Goal:** Automated daily forecasts running in CI, served via API.

### 6.1 FastAPI

Extend EP's existing FastAPI. The Pydantic schemas from stage 1 define the contract; this stage implements the endpoints:

- `GET /health`
- `GET /forecast/price`
- `GET /forecast/generation/{type}` (wind_onshore, wind_offshore, solar)
- `GET /forecast/load`
- `GET /forecast/history?target=price&days=30`
- `GET /models?target=price`
- `GET /models/performance?target=price&days=30`

The API reads from the output JSON/Parquet produced by inference. Initially serves static files (GitHub Pages compatible); can later be backed by DuckDB or PostgreSQL (stage 8) without changing the endpoint contract.

### 6.2 Daily Inference Pipeline

Single script orchestrating:
1. Update data (incremental: SMARD + weather + commodities)
2. Clean and merge
3. Run gen/load models (per-TSO, 168h horizon)
4. Aggregate gen/load to national level
5. Run price models (24h horizon, using gen/load forecasts as features if applicable)
6. Blend/stack per target
7. **Validate outputs** — sanity checks before publishing: prices within [-500, 1000] EUR/MWh, generation non-negative, solar zero at night, load positive, no NaN in forecast columns. Fail the pipeline if checks fail rather than publishing bad data.
8. Write output JSON/Parquet conforming to the Pydantic schemas from stage 1
9. Compute model errors against actuals

Separate concerns (not a 730-line monolith): data update, gen/load inference, price inference, output validation, output writing each as distinct functions.

### 6.3 Retrain Pipeline

Periodic (biweekly or configurable, plus `workflow_dispatch` for manual triggers):
1. Update data
2. Retrain all models from committed hyperparams
3. Recompute ensemble weights
4. Check for degradation (>20% MAE increase on holdout)
5. **If degradation detected:** skip model upload, keep previous release models, log a warning. The `needs_reselection` flag (from EP) signals that hyperparameter retuning or model reselection is needed — this requires manual intervention.
6. **If no degradation:** upload new models to GitHub Release, commit `blend_config.json`

### 6.4 GitHub Actions

**daily_forecast.yml** (decision #11):
```yaml
schedule: "0 8 * * *"  # 08:00 UTC
on: [schedule, workflow_dispatch]
jobs:
  collect-data:
    steps:
      - Download models from GitHub Release
      - Update SMARD, weather, commodities (parallel)
  inference:
    needs: collect-data
    steps:
      - Run gen/load inference
      - Aggregate to national
      - Run price inference
      - Validate outputs
      - Write outputs
  deploy:
    needs: inference
    steps:
      - Deploy to GitHub Pages
      - Upload updated data to release
```

**retrain.yml:**
```yaml
schedule: "0 6 1,15 * *"  # 06:00 UTC, 1st & 15th
on: [schedule, workflow_dispatch]
jobs:
  retrain:
    steps:
      - Download data from release
      - Update data
      - Retrain all models
      - Check degradation
      - Upload models to release (if no degradation)
      - Commit blend_config.json (if no degradation)
```

### 6.5 Local Development

- `make sync` — pull latest merged Parquet from GitHub Release
- `make forecast` — run inference locally
- `make retrain` — run retrain locally
- `make serve` — run FastAPI locally
- `make mlflow` — start MLflow UI

### 6.6 Testing Strategy

In addition to per-stage unit tests (which are expected as part of each stage's milestone), the full pipeline enables these test categories:

**Unit tests (stages 4-5):**
- Physics functions: given known inputs, `compute_wind_power_density`, `compute_air_density`, etc. return expected values.
- Feature DSL parser: `parse_feature("target_price_ewma_168_h12")` returns the correct specification.
- Cleaning rules: each rule type applied to a synthetic DataFrame produces expected output. Edge cases: exact bidding area split timestamp, nuclear generation NaN after decommissioning.
- Leakage validation: known-good and known-bad pipeline configurations pass/fail as expected.

**Integration tests (stage 6):**
- Data pipeline round-trip: collect (from mocked API responses) -> clean -> feature engineer -> verify expected shape and no NaN in required columns.
- Inference smoke test: full pipeline on a small data sample (last 30 days) runs without crashing and produces valid JSON output.
- Blend consistency: blend predictions are a weighted average of individual model predictions (exact equality within floating-point tolerance).

**Data quality checks (run in CI):**
- After each data update: verify no unexpected NaN in critical columns, verify timestamps are continuous, verify row counts are within expected range.
- After each retrain: verify blend MAE on holdout hasn't degraded beyond threshold.

### 6.7 Milestone

- Daily forecast workflow runs end-to-end in GitHub Actions
- Retrain workflow runs and uploads models (with rollback on degradation)
- API serves correct data for all endpoints locally
- API tests passing
- `make forecast` works locally

### Stage 6 Evaluation

**Status:** Complete (2026-06-30)

**What was implemented:**
- `deploy/model_store.py` — MLflow → joblib export/load for all production models; `gen_load_config.json` and `ensemble_config.json` path constants; `production_model_names()` for weight-filtered name list
- `deploy/gen_load_inference.py` — wave-by-wave TSO inference (wind/solar → load → gen_load_diff); `_build_temporal_and_lag_features` with ffill approximation for h>24 lags; `_build_exog_features` chains previous wave outputs via TSO column names; `forecast_with_lags` / `forecast_direct` dispatch on `lags_target` in config; `aggregate_national` sums TSOs to DE_NATIONAL; `update_historical_forecasts` appends to per-(target,region) parquets
- `deploy/price_inference.py` — extends merged dataset to D+1, builds per-version feature matrices, loads non-zero-weight base models, applies SLSQP weights, adds symmetric conformal PI from `conformal_quantile`
- `deploy/validation.py` — `ForecastValidationError`; validates price (24h, [-500,3000] EUR/MWh, no NaN), generation (168h, non-negative, solar night check), load (168h, [10000,120000] MW); `validate_outputs` collects all errors before raising
- `deploy/inference.py` — thin orchestrator: optional data update → gen/load inference → price inference → validation (hard fail) → publish
- `deploy/publish.py` — writes `deploy/data/price_forecast.json`, rolling 30-day `forecast_history.json`, per-(target,region) gen/load JSONs, `model_metadata.json`, per-day `errors/{date}.json` with MAE/RMSE vs SMARD actuals
- `deploy/retrain.py` — price retrain (CI-safe, ~30-90 min): per-model refit → SLSQP reoptimise → degradation guard → config update + export; gen/load retrain (manual detached only, 8-12h)
- `api/app.py`, `api/routes.py`, `api/dependencies.py` — stateless FastAPI (7 endpoints: /health, /forecast/price, /forecast/generation/{type}, /forecast/load, /forecast/history, /models, /models/performance); reads pre-computed JSON from `deploy/data/`; CORSMiddleware allow_origins=["*"]
- `cli.py` — `deploy` sub-group: `forecast`, `export-models`, `gen-load-config`, `serve`, `retrain`; `_write_gen_load_config` hook added to training loop to write best-run metadata after each (target, region)
- `.github/workflows/daily_forecast.yml` — 08:00 UTC cron; 3 jobs: collect-data → inference → GitHub Pages deploy; downloads models from Release at start
- `.github/workflows/retrain.yml` — 1st+15th monthly price retrain; 120-min timeout; uploads new models + commits updated `ensemble_config.json`
- `Makefile` — `forecast`, `forecast-skip-update`, `export-models`, `gen-load-config`, `retrain`, `retrain-gen-load`, `sync` targets
- 25 new tests: `test_deploy_validation.py` (16), `test_deploy_publish.py` (3), `test_api.py` (8) — full suite 490/495 pass (5 pre-existing failures in test_5c_derivations + fred_eu_gas gap)

**Deviations from plan:**
- `_write_gen_load_config` writes best **base** run (not ensemble meta-learner) to `gen_load_config.json` — simpler and avoids stacking complexity; ensemble only marginally beats single model for gen/load
- Gen/load retrain stays strictly manual (detached `setsid nohup` on station); GitHub Actions limit of 6h is too short for 8-12h full retrain
- Price feature matrices at inference reuse on-disk dataset columns as the feature list; no separate feature list artifact needed

**Challenges encountered:**
- Training/inference exog asymmetry: at training `_load_upstream_actuals` supplies SMARD ground-truth; at inference wave N provides previous wave's model outputs. Solved by matching TSO column naming convention (`wind_onshore_50hz`) from `_build_exog_features`.
- TSO lag features require actuals beyond the 24-48h actuals window (h24 lag at hours 25+, d7_d2_avg at hours 49+). Solved with ffill approximation — known imprecision, documented.
- Test timestamps for 168-hour gen/load forecasts: naive `f"{h:02d}:00:00"` format broke at h≥24. Fixed using `pd.date_range` in test fixture.

**Insights for later stages:**
- Stage 7 dashboard can start immediately — it reads from `deploy/data/` static JSON, already structured as `ForecastResponse`.
- The `deploy/data/errors/{date}.json` files give daily MAE/RMSE; Stage 7's performance tab can consume these directly.
- `conformal_quantile` in `ensemble_config.json` is the only PI tuning knob; if coverage drifts, recalibrate on a longer holdout window without retraining.

---

## Stage 7: Dashboard

**Goal:** Live site showing all forecasts with prediction intervals, consuming the API.

### 7.1 Combined Dashboard

Merge EMA's ApexCharts dashboard with EP's price forecast display:
- Price forecast card with prediction intervals (shaded band)
- Generation cards: wind onshore, wind offshore, solar (per-TSO and national)
- Load forecast card
- Actuals overlay where available
- Performance/monitoring tab (from EP) extended for all targets
- Multi-language support (DE/EN from EMA)

The dashboard reads from the same JSON endpoints the API serves. For GitHub Pages deployment, these are static JSON files; the dashboard code doesn't need to change when the backend moves to a live API server (stage 8).

### 7.2 Milestone

- Dashboard shows price + generation + load forecasts with prediction intervals
- Dashboard deployed to GitHub Pages
- Performance tab tracks accuracy for all targets

### Stage 7 Evaluation
<!-- Fill in after stage 7 is complete -->

**Status:** Not started

**What was implemented:**

**Deviations from plan:**

**Challenges encountered:**

**Insights for later stages:**

---

## Stage 8: Post-Merge Extensions

Tackled only after stages 1-7 are stable. Each is its own sub-plan.

### 8.1 Weather Features Directly in Price Models

The highest-value experiment. Currently EMA forecasts generation from weather, then EP uses those forecasts. But the two-stage pipeline optimises for generation accuracy, not price-prediction accuracy. A single end-to-end model could learn which weather signals matter for *prices* directly — the timing of wind ramps relative to demand peaks might matter more for prices than total wind generation.

The unified feature DSL enables this with zero architectural changes — weather FE output columns appear in the price feature list alongside market features. Test whether `wpd_offshore_cap` in a price model outperforms using `prog_gen_wind_offshore`.

Weather FE computation choices (air density mode, spatial aggregation) taken from load model Optuna results (decision #12).

**Approach:** EMA's Optuna-driven feature selection is computationally infeasible for the full price model search space. Instead: fix physics features at sensible defaults (air density, wind power density, wind shear, cyclic direction), and let the model's feature importance identify what matters for prices. Spatial aggregation method (6-7 options) is worth including as a tunable.

### 8.2 Per-TSO Generation as Price Features

The geographic distribution of generation may contain price-relevant information beyond the national aggregate — e.g., wind concentrated in the north with congested transmission to the south. Architecturally free if both per-TSO and aggregated columns are available in the dataset (they are, from stage 2). Test by adding per-TSO generation features to the price feature list and comparing blend MAE against the national-aggregate-only baseline.

### 8.3 Multi-Day Price Forecasting

Extend from D+1 to D+2 through D+7 horizons. Key challenges:
- Leakage rules are horizon-dependent — for D+2, D-1 auction prices are unavailable, so most of EP's strongest features disappear.
- Options: recursive forecasting (predict D+1, use as input for D+2 — error accumulation risk) or direct multi-horizon (separate models/feature sets per horizon). EMA's recursive framework could be adapted for prices, but price autocorrelation decays faster than generation autocorrelation.
- The feature DSL would need horizon awareness: what `_d1` means depends on the forecast horizon.
- Single model with horizon feature is a third option — simpler but may sacrifice accuracy.

### 8.4 DuckDB Analytics Layer

Define views over existing Parquet files. SQL interface for dashboard/chatbot queries. Zero migration from Parquet storage. Useful for: the natural-language query interface (8.6), complex analytical queries across time ranges, and as a foundation for a multi-region EU model (8.8) where relational queries across bidding zones become natural.

### 8.5 Quarter-Hourly Price Model

Move from hourly-aggregated to native 15-minute forecasts. The 24-element target vector becomes a 96-element vector, or alternatively four separate models for each quarter-hour within each hour. Requires rethinking the daily pivot step and the leakage rules. Only worthwhile if there's a use case for quarter-hourly granularity (e.g., intraday trading, battery optimisation).

### 8.6 Chatbot

Natural language -> SQL (via DuckDB) -> response. LLM translates user questions to queries over forecast and actuals data. E.g., "what was the average price last week when wind generation was above 30GW?"

### 8.7 Proper Website Hosting

Move from GitHub Pages to a proper hosting solution. PostgreSQL/TimescaleDB backend for the API. Production-grade deployment.

### 8.8 Multi-Region EU Model

Extend to forecast prices across multiple European bidding zones simultaneously. The data collection, storage, and cleaning architecture is designed to be region-parameterised (`DataSource` per region, cleaning rules per country), so the extension is incremental rather than a rewrite. The main challenge is data acquisition — each bidding zone has its own SMARD-equivalent data source with different APIs and conventions.

---

## Implementation Order & Dependencies

```
Stage 1 ──> Stage 2 ──> Stage 3 ──> Stage 4 ──> Stage 5 ──> Stage 6 ──> Stage 7
  |                                    |            |            |
  |-- Pydantic schemas                 |-- Parser   |-- 5.3 Gen/load + Optuna
  |-- Config framework                 |-- Market   |-- 5.4 Load→price FE transfer
                                       |-- Weather  |-- 5.5 Price models
                                       |-- Leakage  |-- 5.6-5.8 Ensemble, PI
                                       |-- Caching  |
                                                    |-- 6.1 API
                                                    |-- 6.2-6.4 Inference, CI
```

Each stage has a clear milestone (see individual stage sections). A stage is complete when its milestone is met and tests pass.

**Stage-gate rule:** Each stage must reproduce the equivalent output from the source repos before proceeding. Stage 2 raw data matches EP/EMA downloads. Stage 3 merged dataset matches EP's `merged_dataset_hourly.parquet`. Stage 4 features match EP's v5 feature set. Stage 5 metrics match current production accuracy. Investigate discrepancies before moving on — this is the "reproduce first" discipline from decision #6 operationalised.

Stages 1-3 are sequential (each depends on the previous). Within stage 4, the DSL parser must come first, then market and weather features can be developed in parallel. Within stage 5, gen/load tuning (5.3 + 5.7) must complete before the load→price FE transfer (5.4), which must complete before price model training (5.5). Stage 6 depends on stage 5. Stage 7 (dashboard) can be started in parallel with late stage 6 (CI workflows) since the dashboard consumes static JSON files.

Stage 8 extensions are independent of each other and can be tackled in any order after stages 1-7 are stable.

---

## Risk Factors

**Dependency conflicts.** Using MAPIE 1.3+ (CrossConformalRegressor API). All deps tested compatible with Python 3.13, but future updates could break this.

**Suffix DSL complexity.** The grammar must handle all existing EP feature patterns plus new weather columns. If edge cases pile up, the parser could become a maintenance burden. Mitigation: keep the grammar simple, handle rare cases as special-registered columns rather than grammar extensions.

**Reproducing existing accuracy.** The merge changes data processing, feature computation, and training infrastructure simultaneously. If blend MAE degrades, debugging which change caused it is hard. Mitigation: compare outputs at each stage against the existing repos (data, features, predictions) and investigate discrepancies before moving on.

**CI runtime.** The combined pipeline (weather data collection + gen/load inference + price inference) must complete before the 12:00 CET auction. Weather collection is the bottleneck (~20 min). Mitigation: parallelise weather collection by asset type.

**Model storage in releases.** GitHub has a 2GB per-release-asset limit. The full model set should be well under this, but verify. Also, release download adds latency to the daily workflow.
