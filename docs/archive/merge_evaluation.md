# Energy Forecasting Repo Merge — Evaluation Report

Comprehensive stage-by-stage comparison of `energy_prices` (EP) and `energy_market_analysis` (EMA) for merging into the unified `energy-forecasting` repo.

**Date:** 2026-03-26

---

## Source Repos

**energy_prices (EP):** Day-ahead DE-LU electricity price forecasting. Hourly global models, 8-model inverse-MAE blend ensemble, MLflow tracking, FastAPI, biweekly automated retrain. Python 3.13.5. ~12,700 lines.

**energy_market_analysis (EMA):** Weather-based generation/load forecasting per TSO (50Hz, Amprion, TenneT, TransnetBW, Creos). 168-hour horizon, recursive multi-step forecasting, MAPIE conformal prediction intervals, stacking ensembles. Fork of upstream project. Python 3.11.5. ~38 Python files.

**Planned changes during/after merge:**
- Combine load, generation, and price forecasts into a single site
- Use weather features directly in price models (removing two-stage process)
- Extend price models further into the future
- Energy dashboard, forecasting API, chatbot, proper website (lower priority)

---

## Stage 1: Data Collection

### EP

**Sources:**
- **SMARD API** — National-level hourly generation, consumption, cross-border flows, prices for DE-LU and DE-AT-LU. 44 filter keys. Parallel download with `ThreadPoolExecutor(max_workers=10)`.
- **ICAP Carbon Action** — EU carbon allowances (official, ~2-month lag). Two-phase download (Phase 3: 2014-2018, Phase 4: 2019+).
- **Yahoo Finance** — TTF gas futures, Brent crude, CO2.L carbon proxy.
- **FRED** — Historical gas price reconstruction (EU monthly + US Henry Hub daily) for Dec 2014 - Oct 2017 gap.
- **Energy Charts** — Day-ahead prices as secondary/validation source.
- **EMA overlay** — Fetches generation/load forecasts from EMA's GitHub Pages output.

**Architecture:** Clean abstract `DataSource` base class with shared `update()` method handling redundancy window, overlap trimming, deduplication. Each source implements `_fetch_update()`. Typer CLI orchestrates. Bisect-based timestamp matching for precise incremental updates.

**Strengths:**
- Dual-source commodity strategy with bias correction and validation (carbon: r=0.9857, TTF: r=0.9656)
- Clean OOP abstraction eliminates duplication across 5 source types
- Incremental update logic with configurable redundancy window (14 days default)
- Bootstrap logic for missing keys (auto-fetch last 45 days)

**Weaknesses:**
- No weather data — relies entirely on EMA overlay via fragile HTTP/local path bridge
- National-level SMARD only (no per-TSO granularity)
- Saves SMARD as CSV first, then converts to Parquet (unnecessary intermediate)
- No physical validation on incoming data
- No retry/backoff on SMARD API calls

### EMA

**Sources:**
- **SMARD v2 API** — Per-TSO generation and load for 4 German TSOs + Luxembourg (Creos). 12 filter categories x 5 TSOs = 60 combinations minus 8 known-missing. Crash-resilient per-column saves.
- **Open-Meteo** — Weather from 30+ curated locations across three API endpoints:
  - Archive API (historical actuals, 2015+)
  - Historical Forecast API (past forecasts as issued, ~2022+)
  - Forecast API (current 14-day ahead)
  - 16 hourly variables: temperature, humidity, pressure, precipitation, cloud cover, wind speed/direction (10m/100m), wind gusts, 6 radiation variables
- **SMARD national** — Aggregated national data (legacy collector)
- **EPEX SPOT** — Day-ahead prices from external CSV files

**Architecture:** Standalone collector scripts per source with duplicated update patterns. `ParquetOperations` class for efficient I/O (zstd compression level 4, dtype downcasting float64->float32). Location database in `eu_locations.py` with lat/lon, installed capacity, roughness lengths, hub heights, TSO mappings.

**Strengths:**
- Per-TSO SMARD data is strictly more informative than national
- Weather collection is the crown jewel: physics-aware location selection, capacity-weighted, three-endpoint strategy distinguishing actual vs. forecast weather
- 30+ curated locations with rich metadata (a significant domain knowledge asset)
- Physical bounds validation on weather data (temperature -45-50C, wind 0-200 km/h, etc.)
- Crash-resilient saves (after each column), retry with 20s backoff
- No API keys required (all public APIs)
- Solar elevation/azimuth computed via pysolar at collection time (expensive, deterministic, done once)

**Weaknesses:**
- No commodity data (carbon, gas, oil)
- No abstract base class — each collector is standalone with duplicated patterns
- Pipenv-based (less modern than pyproject.toml)
- Sequential fetching for SMARD v2 (no parallelism)

### Recommendation

| Component | Source | Changes Needed |
|-----------|--------|---------------|
| **Architecture** | EP | Extend `DataSource` abstraction for weather and per-TSO sources |
| **SMARD data** | EMA | Per-TSO granularity; refactor into `SmardV2Source(DataSource)` |
| **Weather data** | EMA | Take wholesale; wrap in `OpenMeteoSource(DataSource)` preserving three-endpoint logic |
| **Location metadata** | EMA | Take `eu_locations.py` as curated asset |
| **Commodity data** | EP | Take dual-source strategy wholesale; evaluate Brent (54% missing) |
| **Energy Charts** | EP | Keep as price fallback |
| **I/O format** | EMA | Parquet from the start with zstd + dtype downcasting; skip CSV intermediate |
| **Incremental updates** | Merge | EP's bisect-based redundancy + EMA's `combine_first()` non-destructive merge |
| **CLI** | EP | Extend Typer CLI with new source commands |

**Key insight:** The merge eliminates the fragile bridge where EP fetches forecasts from EMA's published output. Weather data feeds directly into all models.

---

## Stage 2: Data Storage

### EP
- **Raw:** CSV per SMARD measure key -> Parquet (combine step)
- **Interim:** Wide-format Parquet (one column per measure)
- **Processed:** Merged Parquet with regime indicators and commodity data
- **Features:** Pipeline-transformed Parquet cached in MLflow (versioned by `run_id`)
- **Models:** MLflow SQLite DB (`mlflow.db`) + joblib files in `models/production/`
- **Deploy:** Static JSON on GitHub Pages
- **Data persistence:** GitHub Releases for large data (merged_dataset.parquet), GitHub Actions cache for workflow state

### EMA
- **Raw/Database:** Parquet directly with zstd compression, float32 downcasting
- **Models:** Pickle files in `output/{target}/{model}/trained/`
- **Deploy:** Static JSON/CSV on GitHub Pages
- **Data persistence:** Committed to git (database/ directory) — causes repo bloat over time

### Recommendation

| Aspect | Source | Rationale |
|--------|--------|-----------|
| **Raw storage** | EMA | Parquet from the start; skip CSV intermediate |
| **Compression** | EMA | zstd level 4 + dtype downcasting |
| **Experiment tracking** | EP | MLflow (far superior to file-based) |
| **Model serialization** | EP | Joblib preferred over pickle for sklearn models |
| **Data persistence** | EP | GitHub Releases, NOT git commits (prevents repo bloat) |
| **Deploy format** | Merge | JSON for dashboard, Parquet for analytics |

### Storage Backend: Phased Approach

**Phase 1 (during merge):** Parquet files + MLflow SQLite + DuckDB as analytics query layer.
- DuckDB reads Parquet natively (zero-copy), provides SQL interface
- Zero infrastructure change: embedded, no server, works in CI/CD
- Dashboard/chatbot queries become SQL over existing Parquet files
- ~20 lines of code to add

**Phase 2 (when building API/website):** Add PostgreSQL/TimescaleDB for production serving.
- Batch pipeline still writes Parquet
- Sync step loads results into PostgreSQL for concurrent API access
- Separates ML pipeline (batch) from serving layer (concurrent)

---

## Stage 3: Data Cleaning & Preprocessing

### EP

12 sequential, domain-informed rules in `handle_missing_values()`:

1. Drop redundant columns (3 — values already in `target_price`)
2. Nuclear generation zero-fill after decommissioning (April 2023)
3. Austria neighbour flows zero after bidding area split (2018-09-30)
4. Belgium flows zero before reporting start (2017-10-10)
5. Austria direct flows/price zero before split (pre-2018)
6. Norway flows zero before first valid observation
7. Calculate `prognostizierte_erzeugung_sonstige` from `gesamt - wind_und_photovoltaik`
8. Backfill `prognostizierter_verbrauch_gesamt` with actual consumption (r=0.9744)
9. Backfill `prognostizierte_erzeugung_gesamt` with component sum
10. Fill Polish/Swiss prices with `target_price` (zero-spread assumption)
11. Calculate net exports from 14 export/import pairs
12. Cubic spline interpolation for remaining gaps <= 5 consecutive NaNs

Each rule has a documented rationale tied to regulatory events, infrastructure changes, or data reporting patterns. Processing order is deliberate: drop -> structural fills -> calculated fills -> interpolation.

Structural breaks handled via regime indicators: `regime_de_at_lu` (pre/post 2018 split), `regime_quarter_hourly` (post Oct 2025).

**Strengths:** Extremely well-documented domain rules. Regime indicators are clean. Processing order is principled.
**Weaknesses:** Tightly coupled to national-level data. No weather data cleaning. Rules are hardcoded in if-blocks.

### EMA

- `impute_smard_nans()`: Fill with column mean (crude)
- `handle_nans_with_interpolation()`: Linear bidirectional, max gap 48 hours, error on larger gaps
- `fix_broken_periodicity_with_interpolation()`: Detect and fill missing timestamps, max 3 consecutive
- Physical bounds validation on weather variables (hardcoded limits per variable type)
- Derived columns: `gen_load_diff_delu`, `residual_load_[TSO]`

**Strengths:** Physical bounds validation essential for weather data integrity. Periodicity enforcement guarantees continuous hourly series. Per-TSO derived columns useful.
**Weaknesses:** Column-mean imputation is dangerous for time series (ignores temporal structure, creates phantom data post-decommissioning etc.). Less documentation of rationale.

### Recommendation

| Aspect | Source | Rationale |
|--------|--------|-----------|
| **Energy market missing values** | EP | Domain-specific rules far superior to column-mean |
| **Weather data validation** | EMA | Physical bounds checking essential |
| **Weather interpolation** | EMA | Linear interpolation with gap limits appropriate |
| **Structural breaks** | EP | Regime indicators are clean modeling approach |
| **Periodicity enforcement** | EMA | Continuous hourly series guaranteed early |
| **Documentation** | EP | Every rule should have a rationale |

**Architecture improvement:** Config-driven cleaning rules rather than hardcoded if-blocks:

```python
CLEANING_RULES = [
    FillZeroAfter("gen_nuclear", after="2023-04-15", reason="Nuclear decommission"),
    FillZeroAfter("xborder_*_hungary_*", after=BIDDING_AREA_SPLIT, reason="DE-AT-LU split"),
    PhysicalBounds("temperature_2m_*", min=-45, max=50, reason="Physical limits"),
    LinearInterpolate("*", max_gap=48, reason="Small gap fill"),
]
```

---

## Stage 4: Feature Engineering

### Fundamental Design Decision: Suffix DSL

Both repos' feature engineering approaches have significant problems:

**EP's sklearn pipeline system** (TransformerMixin, v1-v5 versioning) was over-engineered. The sklearn composability (GridSearchCV, Pipeline) was never actually used, yet it imposed ceremony on every feature. The versioning system (1759 lines in `pipelines.py`, 5 overlapping versions) became a maintenance nightmare.

**EMA's approach** (plain Python classes, Optuna-driven selection) is simpler but tightly coupled to data loading and has no leakage validation.

**Proposed solution: Suffix DSL.** Features are defined as a list of strings. Suffixes encode the transformation. The system parses each suffix and computes the feature.

### Suffix Grammar

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_h{N}` | Hourly lag (exact value N hours back) | `price_h24`, `price_h168` |
| `_d{N}` | Daily average (average of day N ago; STAT defaults to avg) | `price_d1`, `price_d7` |
| `_d{N}_{stat}` | Daily aggregate with explicit stat | `price_d7_std`, `price_d7_max` |
| `_d{N}_d{M}` | Multi-day rolling average (days N to M ago) | `price_d7_d1` |
| `_d{N}_d{M}_{stat}` | Multi-day rolling aggregate | `price_d7_d1_std` |
| `_d{N}_d{M}_h{A}_h{B}_{stat}` | Rolling with hour filter | `price_d7_d1_h8_h20_avg` |
| `_ewma_{span}_d{N}` | EWMA, cutoff end of day N | `price_ewma_24_d1` |
| `_ewma_{span}_d{N}_h{H}` | EWMA, cutoff hour H day N | `price_ewma_186_d1_h10` |
| `{a}__x__{b}` | Interaction term | `price_ewma_6_d1__x__day_index` |
| *(no suffix)* | Raw value (forecasts/static only) | `prog_gen_wind_pv` |

### Short Name Registry

Raw SMARD column names are unwieldy. A short-name registry maps concise names to actual columns:

| Short | Original |
|-------|----------|
| `price` | `target_price` |
| `load` | `stromverbrauch_gesamt_(netzlast)` |
| `gen_wind_on` | `stromerzeugung_wind_onshore` |
| `gen_solar` | `stromerzeugung_photovoltaik` |
| `prog_load` | `prognostizierter_verbrauch_gesamt` |
| `carbon` | `carbon_eur_per_ton` |
| `ttf` | `ttf_eur_per_mwh` |
| `spread_fr` | (derived) target_price - marktpreis_frankreich |
| `wpd_offshore_cap` | (derived) capacity-weighted offshore wind power density |

### How It Works

```python
PRICE_FEATURES = [
    # Temporal
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday", "day_index",
    # Price history (hourly lags for same-hour values, daily for averages)
    "price_h24", "price_h168",
    "price_d1", "price_d7", "price_d14",
    "price_d7_d1", "price_d7_d1_std", "price_d30_d1",
    "price_ewma_6_d1", "price_ewma_24_d1", "price_ewma_2160_d1",
    # Forecasts (no lag — available on prediction day)
    "prog_gen_wind_pv", "prog_load",
    # Commodities (D-2 due to reporting lag)
    "carbon_d2", "ttf_d2", "ttf_ewma_720_d2",
    # Cross-border
    "spread_fr_d1", "net_export_fr_d2",
    # Weather (direct — no two-stage process)
    "wpd_offshore_cap", "temperature_cities_pop",
    # Interactions
    "price_ewma_6_d1__x__day_index",
]

df = engineer_features(raw_df, PRICE_FEATURES)
```

The engine: (1) parses each string into (base_column, derivation, lag_spec), (2) groups by base column to avoid redundant computation, (3) computes derived columns first, (4) applies lags/aggregations, (5) validates against availability rules, (6) returns clean DataFrame with exactly the requested columns.

### Benefits
- Feature list IS the config — no separate YAML, no pipeline objects, no version numbers
- Self-documenting: you can read what `price_ewma_186_d1_h10` means
- Optuna integration: toggle features in/out of the list
- Leakage validation: parse suffix to check implied lag against availability rules
- No sklearn ceremony, no version sprawl

### Leakage Validation (Declarative)

Availability rules declare when data is physically available:

| Column pattern | Max offset | Cutoff hour | Reason |
|---------------|-----------|-------------|--------|
| `price` | D-1 | — | Auction results published D-1 |
| `gen_*` | D-1 | 10 | SMARD publishes by 11am CET |
| `load` | D-1 | 10 | Same as generation |
| `carbon`, `ttf`, `brent` | D-2 | — | Business day delay |
| `prog_*` | D | — | Forecasts published for today |
| `hour_sin`, `is_holiday` etc. | D | — | Deterministic |

The validator parses each feature's suffix and checks the implied lag >= the availability constraint. `price_d1` passes (lag=1 >= offset=1). `price` without suffix would fail (lag=0 < offset=1) unless it's a forecast column.

### EP's Market-Fundamentals Features (preserved)

- Price spreads vs 14 neighbours
- Net exports per country
- Generation mix percentages
- Prognosticated generation/consumption
- Commodity prices with appropriate lags
- Rolling stats, EWMA, same-hour lags
- Temporal features (cyclical, holidays, trend)
- Interaction terms

### EMA's Physics-Informed Features (preserved)

Three specialized classes:

**WeatherWindPowerFE:**
- Air density (dry/moist): rho = P / (R_d * T)
- Wind power density: 0.5 * rho * v^3
- Wind shear: log(v_100m / v_10m)
- Turbulence intensity: rolling_std / rolling_mean
- Gust factor, wind ramp, dew point

**WeatherSolarPowerFE:**
- Cloud/clear sky fractions
- Radiation ratios (direct/diffuse/DNI/global tilted vs shortwave)
- Solar geometry (elevation, azimuth via pysolar)

**WeatherLoadFE:**
- Heating degree hours: max(0, 18 - T)
- Cooling degree hours: max(0, T - 22)
- Wind chill, humidex (thermal comfort)
- Pressure trends, rain indicators, effective solar

**Spatial aggregation:** Multiple strategies (capacity-weighted, inverse-distance, population-weighted). Haversine distance between locations.

**Key EMA design:** Feature engineering choices are Optuna hyperparameters (whether to compute air density, which lag lengths, which aggregation method, whether to drop raw features). This should be preserved.

Weather features don't need temporal lagging — they're forecasts available at prediction time.

### Recommendation

- Implement the suffix DSL as the primary feature engineering interface
- Preserve EP's market-fundamentals features as library functions called by the DSL engine
- Preserve EMA's physics-informed feature classes, called by the DSL for weather-derived base columns
- Preserve EMA's "FE choices as Optuna hyperparameters" pattern
- Implement declarative leakage validation decoupled from any pipeline framework
- Cache processed features in MLflow for reproducibility (keyed by config hash)

---

## Stage 5: Model Training

### EP — Global Hourly Model

Single global model predicts all 24 hours. Each model trained on full hourly dataset (one row per hour, scalar target). `train_and_log()` handles: load features from MLflow -> time-series split -> optional exponential decay sample weights -> wrap in TransformedTargetRegressor -> fit -> evaluate -> log to MLflow.

Models: Ridge/Lasso/ElasticNet, LightGBM, XGBoost, CatBoost. All sklearn-compatible.

**Sample weighting:** Exponential decay with configurable half-life (e.g., 730 days). Formula: `w = exp(ln(2)/half_life * (t - t_max))`. Handles distributional shifts (energy crisis, COVID) without explicit regime modeling.

**Cross-validation:** TimeSeriesSplitter with expanding or sliding window, respecting group boundaries. 5 folds default.

### EMA — Recursive Multi-Step with MAPIE

Each target (per TSO) gets its own model. Recursive forecasting: predict one step, feed prediction back as lagged feature, repeat for 168 hours.

```
for each future hour i in [0, 168):
    set lag features from previous predictions
    predict hour i with MAPIE -> point + 95% CI
    store prediction
```

Models: Same gradient-boosted trees + ElasticNet + Prophet. Each wrapped in MapieRegressor for conformal prediction intervals.

### Comparison

| Aspect | EP (Global Hourly) | EMA (Recursive Multi-Step) |
|--------|-------------------|---------------------------|
| Horizon | 24h (day-ahead) | 168h (7 days) |
| Lag handling | Pre-computed in features | Dynamically updated during recursion |
| Error accumulation | None (direct prediction) | Compounds over 168 steps |
| Uncertainty | None (point predictions only) | MAPIE conformal intervals |
| Inference speed | Fast (one forward pass) | Slow (168 sequential steps) |

### Recommendation

| Aspect | Source | Rationale |
|--------|--------|-----------|
| **Training loop** | EP | `train_and_log()` with MLflow integration (simplified) |
| **Sample weighting** | EP | Exponential decay handles regime shifts elegantly |
| **Recursive forecasting** | EMA | Needed for 168h generation/load forecasts |
| **Direct prediction** | EP | Better for day-ahead price (no error accumulation) |
| **Prediction intervals** | EMA | MAPIE conformal prediction on all models |
| **Multi-step price** | Experimental | Direct vs recursive for extended price horizons — determine empirically |

---

## Stage 6: Hyperparameter Tuning

### EP
Optuna is a dependency but tuning is ad-hoc (notebooks). Best hyperparams committed to `blend_hyperparams.json`. Search spaces not centrally declared. Tuning -> commit -> retrain cycle is manual.

### EMA
Optuna deeply integrated. Centrally declared search spaces in `hyperparameters_for_optuna.py`. Feature engineering choices (air density, lag lengths, spatial aggregation, raw feature dropping) are Optuna hyperparameters. 70-120 trials per model with 3-fold CV. Automated finetune -> train -> forecast pipeline.

### Recommendation

| Aspect | Source | Rationale |
|--------|--------|-----------|
| **Search spaces** | EMA | Centrally declared, per model type |
| **FE as hyperparameters** | EMA | Elegant, effective for weather features |
| **Tuning automation** | EMA | Finetune -> train -> forecast pipeline should be scriptable |
| **Tracking** | EP (improved) | Log Optuna trials to MLflow with proper tags |
| **Committed params** | EP | Saving best params for reproducible retrain is good practice |

The suffix DSL integrates naturally: Optuna toggles items in/out of the feature list.

---

## Stage 7: Ensembling

### EP — Inverse-MAE Weighted Blend

4-stage process:
1. **Select candidates:** Query MLflow, pick top 3 per category (linear/lgbm/xgb/catboost) + 2 random
2. **Validate candidates:** Retrain each with 5-fold sliding-window CV
3. **Select final models:** Best-MAE + best-RMSE per category (8 total)
4. **Train & blend:** Fit on all data minus 90-day holdout, weight = 1/MAE normalized

Current production: 8 models, blend MAE = 9.93 EUR/MWh, R2 = 0.78.

Biweekly retrain with degradation check (>20% MAE increase flags `needs_reselection`).

### EMA — Stacking Ensemble

2-level architecture:
1. **Base models:** LightGBM, XGBoost, ElasticNet — each with CV out-of-fold predictions
2. **Meta-model:** LightGBM/XGBoost trained on OOF predictions + temporal meta-features

Meta-model learns context-dependent weighting (e.g., "LightGBM is better at night").

Can optionally include prediction intervals (lower/upper) as meta-features.

### Comparison

| Aspect | EP Blend | EMA Stacking |
|--------|----------|-------------|
| Combination method | Fixed formula (inverse-MAE) | Learned (meta-model) |
| Adaptiveness | Weights fixed until retrain | Context-dependent weighting |
| Overfitting risk | None (closed-form) | Possible (meta-model) |
| Diversity source | 4 categories x 2 | 2-3 base + 1 meta |
| Prediction intervals | None | Yes (propagated through) |

### Recommendation

Support both — let the target dictate:
- **Price models:** Inverse-MAE blend (EP). More models, simpler combination, proven.
- **Generation/load models:** Stacking (EMA). Fewer models per target, extracts more value.
- **Both approaches:** Which works better for which target is an experimental question.
- **All models:** Add MAPIE prediction intervals. Blend can average intervals; stacking can use them as meta-features.

---

## Stage 8: Prediction Intervals & Uncertainty

### EP
None. Point predictions only.

### EMA
MAPIE conformal prediction on every model:
```python
MapieRegressor(estimator, method='naive', cv='prefit',
               conformity_score=AbsoluteConformityScore(sym=True))
```
- Stores residual quantiles from training data
- Produces symmetric intervals: [y_hat - q95, y_hat + q95]
- Distribution-free, calibrated to ~95% coverage
- Propagated through national aggregation (conservative: sum of bounds)

### Recommendation

Add MAPIE to all models (from EMA). For blend ensembles, use calibrated blend intervals: compute the blend's actual residual distribution on holdout, derive the 95th percentile. This accounts for variance reduction from blending (intervals should be narrower than individual models').

Prediction intervals are valuable for: energy trading risk management, dashboard credibility, model monitoring (widening intervals = declining confidence), chatbot ("how confident is the forecast?").

---

## Stage 9: Deployment & CI/CD

### EP
Two workflows:
- **Daily forecast** (08:00 UTC): data update -> inference -> deploy to GitHub Pages -> upload data to release
- **Biweekly retrain** (06:00 UTC, 1st & 15th): data update -> full retrain -> commit models -> push

Data cached via GitHub Releases (merged_dataset as release asset) + Actions cache. Pre-exported pipelines allow inference without MLflow artifact store.

### EMA
Four chained workflows:
- **collect_data** (07:00 UTC): 5 collection tasks -> commit database/
- **update_forecasts** (triggered): run all models -> commit output/
- **publish_forecasts** (triggered): aggregate + publish -> commit deploy/data/
- **deploy** (triggered): push to gh-pages

3-4 commits per day. Database Parquet files committed to git (causes repo bloat).

### Comparison

| Aspect | EP | EMA |
|--------|------|------|
| Workflows | 2 (independent) | 4 (chained) |
| Commits per day | 0-1 | 3-4 |
| Data persistence | GitHub Releases | Git (repo bloat) |
| Failure isolation | Good | Poor (chain breaks) |

### Recommendation

| Aspect | Source | Rationale |
|--------|--------|-----------|
| **Workflow structure** | EP | 2 workflows is simpler |
| **Data persistence** | EP | GitHub Releases, NOT git commits |
| **Step ordering** | New | Within single workflow: data -> gen/load models -> price models -> deploy |

If using weather directly in price models (not depending on gen/load model output), the ordering constraint relaxes and everything can run in parallel.

---

## Stage 10: Web / API / Dashboard

### EP
FastAPI REST API with 5 endpoints: `/health`, `/forecast`, `/forecast/history`, `/models`, `/models/performance`. Pydantic schemas. CORS enabled. Reads from static JSON files.

Minimal dashboard (GitHub Pages).

### EMA
No API. Full static dashboard: HTML + ApexCharts JS (37KB) + Bootstrap CSS. Multi-language (DE/EN). Interactive charts with zoom/pan. Per-target cards with forecast vs. actuals.

### Recommendation

| Aspect | Source | Rationale |
|--------|--------|-----------|
| **API** | EP | FastAPI + Pydantic, extend for gen/load endpoints |
| **Dashboard** | EMA | ApexCharts-based, polished, multi-language |
| **Architecture** | New | Dashboard calls API; API reads from DuckDB/JSON |

**Phase 1:** Static JSON + GitHub Pages (combine price + gen/load into one dashboard).
**Phase 2:** FastAPI + DuckDB. Dashboard calls API.
**Phase 3:** PostgreSQL, proper hosting, chatbot.

---

## Stage 11: Testing & Monitoring

### EP
11-file pytest suite: data loading, transforms, API endpoints, leakage validation, baselines, metrics, preprocessing, training, temporal features, lag transforms.

Monitoring: blend degradation detection (>20% MAE increase), per-model error tracking, retrain history.

### EMA
2-file pytest suite (minimal): data loaders, export.

Monitoring: prediction interval coverage tracking. No automated degradation detection.

### Recommendation
- Extend EP's test structure with EMA's PI coverage tests
- Leakage validation tests critical — extend for weather features
- Add integration tests for end-to-end pipeline
- Monitoring: degradation detection (EP) + PI width tracking (EMA) + data freshness checks

---

## Stage 12: MLflow Conventions

### Problems in EP
- Datasets tracked as models (wrong abstraction)
- No enforced metadata conventions
- Too many experiments (annoying to navigate between them)
- No good cross-experiment comparison

### Proposed Design

**Minimal experiments.** One experiment per target (e.g., `price`, `wind_onshore`, `load`). All run types (datasets, tuning, training, production, inference) in the same experiment. Use tags for filtering, not experiment boundaries.

**Required tags (enforced by validation):**

```python
REQUIRED_TAGS = {
    "run_type": str,    # "dataset" | "tuning" | "training" | "production" | "inference"
    "target": str,      # "price" | "wind_onshore_50hz" | "load_tenn" | ...
    "model_type": str,  # "lgbm" | "xgboost" | "catboost" | "linear" | "blend" | "dataset"
}

REQUIRED_BY_TYPE = {
    "dataset":    ["feature_count", "config_hash", "row_count"],
    "tuning":     ["n_trials", "dataset_run_id"],
    "training":   ["dataset_run_id", "feature_list_hash"],
    "production": ["n_models", "holdout_days"],
    "inference":  ["forecast_date"],
}
```

Filter in UI: `tags.run_type = "training"` to see all training runs, `tags.model_type = "lgbm"` for all LightGBM runs.

**Validation wrapper** that checks required tags before closing a run — can't log incomplete runs.

**Dataset tracking:** Use MLflow's `mlflow.log_input()` properly (not the model registry). Training runs reference datasets via `dataset_run_id` tag.

---

## Summary: What to Take from Each Repo

| Stage | EP | EMA | New |
|-------|------|------|------|
| Data collection | `DataSource` abstraction, commodities | Per-TSO SMARD, weather, locations | Unified source classes |
| Storage | MLflow, GitHub Releases | Parquet + zstd + float32 | DuckDB analytics layer |
| Cleaning | Domain-informed rules, regimes | Physical bounds, periodicity | Config-driven rules |
| Feature engineering | Market features, leakage validation | Physics features, FE as hyperparams | **Suffix DSL** |
| Model training | train_and_log, sample weighting | Recursive multi-step, MAPIE | Both, target-dependent |
| Hyperparameter tuning | MLflow logging | Optuna integration, central search spaces | Combine: Optuna -> MLflow |
| Ensembling | Inverse-MAE blend (8 models) | Stacking meta-learner | Both (experimental) |
| Uncertainty | None | MAPIE conformal prediction | MAPIE on everything |
| Deployment | 2 workflows, releases | 4 chained workflows | EP's 2-workflow structure |
| Dashboard/API | FastAPI | ApexCharts dashboard | Combined |
| Testing | 11-file pytest | Minimal (2 files) | Extend EP's suite |
| Monitoring | Degradation detection | PI coverage tracking | Both |
| MLflow | Fragmented conventions | File-based (no MLflow) | Enforced conventions |

---

## Open Experimental Questions

These should be determined empirically rather than decided upfront:

1. **Direct vs. recursive forecasting for extended price horizons.** Direct multi-horizon avoids error accumulation but needs more thought about feature availability at each horizon. Recursive is simpler to implement.

2. **Inverse-MAE blend vs. stacking ensemble for each target type.** Blend is simpler and proven for prices. Stacking may extract more from fewer models for gen/load. Test both.

3. **Weather features directly in price models vs. two-stage.** The merge enables this. Does `wind_power_density_offshore` in a price model outperform using `prog_gen_wind_offshore` (the output of a separate generation model)?

4. **Per-TSO vs. national features for price models.** Per-TSO weather is essential for gen/load. But does per-TSO granularity improve price forecasting, or is national aggregation sufficient?

5. **Optimal feature set per target.** The suffix DSL + Optuna makes this experimentally tractable.
