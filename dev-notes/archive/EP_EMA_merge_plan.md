# EP + EMA Merge Plan

Analysis and recommendations for combining the `energy-prices` (EP) and `energy_market_analysis` (EMA) repositories into a single energy forecasting repo.

**EP** forecasts day-ahead electricity prices for the DE-LU bidding area using gradient-boosted tree ensembles with market fundamentals, cross-border flows, and commodity prices.

**EMA** forecasts weather-based generation and load for the German market, broken down by TSO region. It is a fork of [vsevolodnedora/energy_market_analysis](https://github.com/vsevolodnedora/energy_market_analysis). EMA currently serves as a feature provider for EP — its generation/load forecasts replace SMARD's TSO forecasts (which are published too late to be useful for day-ahead trading).

**Why merge:** The two repos are tightly coupled — EMA produces features that EP consumes via a fragile bridge (fetching CSVs from GitHub Pages or hardcoded local paths). A single repo eliminates this fragility, enables experiments that span both domains, and presents a unified project for portfolio purposes.

**Scope of this plan:** Reproduce existing functionality from both repos in a single, cleaner codebase. Experimental extensions (multi-day forecasting, weather features directly in the price model, API, etc.) are listed in section 11 but are explicitly out of scope for the initial merge.

---

## 1. Data Collection

### Current state

**EP** collects market-focused data from five sources, each implemented as a `DataSource` subclass with a generic `update()` method:

- **SMARD API** — generation by source, consumption, cross-border flows, market prices for DE-LU and DE-AT-LU bidding zones. Individual CSVs per measure key. Parallel downloads via `ThreadPoolExecutor`.
- **ICAP Carbon Action** — EU carbon allowances with ~2 month publication lag.
- **Yahoo Finance** — TTF gas futures, Brent crude, CO2.L equity proxy.
- **FRED** — historical gas price reconstruction (EU monthly + US Henry Hub daily).
- **Energy Charts** — day-ahead prices as secondary/validation source.

**EMA** collects weather and per-TSO energy data:

- **SMARD v2** — per-TSO generation and load actuals (50Hertz, Amprion, TenneT, TransnetBW, Creos). Single wide parquet file with column-by-column crash-resilient saving.
- **Open-Meteo** — weather data from 30+ curated locations near actual generation assets. Three separate API endpoints (historical actuals, past forecasts, future forecasts) stitched into continuous time series. Retry logic with exponential backoff.
- **EPEX SPOT** — day-ahead and intraday auction prices from scraped files.

### EP strengths
- Clean OOP abstraction via `DataSource` base class — generic update flow (load → overlap → fetch → trim → concat → save) reduces duplication.
- Unified CLI via Typer with `Makefile` orchestration.
- Bootstrap logic for missing CSVs — new SMARD measures auto-fetch last 45 days.
- Good redundancy handling (14-day re-fetch window, `keep="last"` deduplication).

### EP weaknesses
- Stores raw SMARD data as many individual CSVs — inefficient for downstream joining.
- No physical validation on incoming data.
- No retry/backoff on SMARD API calls (raw `requests.get` with no timeout on `get_data()`).
- No weather data collection — relies entirely on EMA.

### EMA strengths
- Parquet storage from the start — SMARD v2 saves directly to wide parquet.
- Physics-based validation bounds on weather data (`phys_limits` dict).
- Crash-resilient incremental updates — saves after each column.
- Rich weather data with curated location metadata (30+ lat/lon coordinates, TSO mappings, installed capacity data).
- Retry logic with backoff for Open-Meteo.
- Combines three Open-Meteo API endpoints into continuous time series.

### EMA weaknesses
- No OOP abstraction for data sources — each collector is standalone with its own conventions.
- No commodity data collection.
- Location metadata in a 1,800-line Python file (`eu_locations.py`) mixing data with code.
- `sys.argv` parsing instead of a proper CLI.

### Recommendations

- **Extend EP's `DataSource` abstraction** to cover weather and per-TSO data as new source classes (`OpenMeteoSource`, per-TSO `SmardSource`).
- **Unify SMARD collectors.** EMA's v1 and v2 are not different APIs — they both hit `smard.api.proxy.bund.dev` with different region codes. EP's `SmardSource` already parameterises on region/resolution. Extend it to accept TSO region codes alongside bidding-zone codes. Port EMA's `FILTER_COLUMNS`, `TSO_SUFFIX`, and `KNOWN_MISSING` mappings as config.
- **Adopt parquet-from-the-start** everywhere. Drop the raw CSV stage for SMARD.
- **Add retry/backoff trivially** (a `requests.Session` with `urllib3.util.Retry` — ~3 lines) but don't architect around it. The SMARD API has never failed, so this is defensive rather than critical.
- **Extract `eu_locations.py`** into a JSON or YAML config file.
- **Unify CLI** using Typer with EMA's task-based structure.

---

## 2. Storage

### Current state

**EP** uses a two-stage approach: raw CSV → interim parquet (wide format) → processed parquet. Commodities and SMARD are kept as separate parquet files until feature engineering. Paths managed centrally via `get_path()` and `RAW_DIRS`. Default parquet settings (no compression, no dtype reduction).

**EMA** uses parquet from the start. SMARD v2 → single wide parquet. Weather → parquet per TSO per asset type in a `database/{country}/{source}/{asset_type}/{TSO}/` hierarchy. Three weather parquet files per directory: `history_hourly.parquet`, `hist_forecast_hourly.parquet`, `forecast_hourly.parquet`. `ParquetOperations` class applies zstd compression (level 4) and automatic dtype reduction (float64→float32, int64→int32).

### Recommendations

- **Adopt parquet-from-the-start** everywhere. Skip EP's raw CSV stage entirely.
- **Use EMA's compression/dtype reduction** but simplify to a utility function: `save_parquet(df, path, compress=True, reduce_dtypes=True)`.
- **Unified directory layout:**
  ```
  data/
  ├── raw/
  │   ├── smard/{region}/history.parquet
  │   ├── weather/{asset_type}/{tso}/history.parquet
  │   ├── commodities/{source}.parquet
  │   └── energy_charts/da_price.parquet
  ├── interim/
  │   └── merged.parquet
  └── processed/
      └── features.parquet
  ```
- **Keep commodities and SMARD as separate files** until feature engineering — EP's rationale (independent update cycles, experimentation flexibility) is sound.
- **Eliminate the cross-repo bridge.** Weather-based forecasts flow directly into feature engineering rather than being fetched from GitHub Pages.
- **Use parquet for now.** DuckDB or a traditional database can be layered on later if needed (see section 11).

---

## 3. Cleaning & Missing Value Handling

### Current state

**EP** handles structural missingness with 12 domain-specific rules applied in fixed order by `handle_missing_values()`. Rules include: nuclear zeroed after decommissioning, cross-border flows zeroed before/after bidding area split, forecast columns filled from actuals, Poland/Switzerland prices filled with target price (zero-spread assumption), net exports recalculated, and cubic spline interpolation for small gaps (≤5 consecutive NaNs). Rules documented in `processing.py` config and `DATA.md`. DST transitions handled explicitly (interpolate spring-forward, average fall-back). The function is wrapped as an sklearn `MissingValueHandler` transformer.

**EMA** handles weather data quality with physics-based validation (physical limit bounds for all weather variables), z-score outlier detection on target columns (3σ threshold), gap validation with configurable max gap, linear interpolation, periodicity enforcement, and mean-fill for SMARD reporting gaps. Hard NaN gate before training: `check_for_nans_and_raise_error()` crashes the pipeline if any NaN remains.

### Assessment

EP's approach is rule-based and domain-driven — every rule has a documented rationale. EMA's approach is statistical and fail-fast. Neither handles the other's problems: EP has no physical validation of incoming data; EMA has no concept of structural breaks or regime changes.

### Recommendations

**Layer the cleaning:** physical bounds validation first (from EMA) → structural/domain rules (from EP) → statistical gap-filling last.

**Architecture: self-explanatory config → generic interpreter → functions.**

Make EP's structural rules data-driven via YAML config:

```yaml
structural_fills:
  - columns: ["stromerzeugung_kernenergie"]
    action: fill_zero_after_last_valid
    reason: "German nuclear decommissioned April 2023"

  - columns:
      - "cross-border_flows_hungary_exports"
      - "cross-border_flows_hungary_imports"
      - "cross-border_flows_slovenia_exports"
      - "cross-border_flows_slovenia_imports"
    action: fill_zero_after
    date: "2018-09-30T22:00:00Z"
    reason: "Austria neighbor flows irrelevant after DE-AT-LU split"

  - columns:
      - "cross-border_flows_austria_exports"
      - "cross-border_flows_austria_imports"
      - "marktpreis_österreich"
    action: fill_zero_before
    date: "2018-09-30T22:00:00Z"
    reason: "No separate Austria flows before bidding area split"

calculated_fills:
  - column: "prognostizierte_erzeugung_sonstige"
    action: fill_from_difference
    source_total: "prognostizierte_erzeugung_gesamt"
    source_subtract: "prognostizierte_erzeugung_wind_und_photovoltaik"

physical_limits:
  temperature_2m: [-45, 50]
  wind_speed_10m: [0, 200]
  target_price: [-500, 1000]
  load: [0, null]

interpolation:
  method: cubicspline
  max_gap: 5
  exclude: ["regime_de_at_lu", "regime_quarter_hourly", "target_price"]
```

A single generic function interprets the config. Each `apply_*` function (~20-40 lines) reads the action type and dispatches. Domain knowledge lives entirely in the YAML. Adding new rules is one YAML entry, not editing a procedural function. The YAML's `reason` field doubles as documentation.

For the 2-3 rules requiring complex conditional logic (e.g., EP's step 9 with a 30-day recency check and fallback chain), use named "special" handlers that the config references.

**Additional cleaning recommendations:**
- Replace EMA's mean-fill for SMARD gaps with time-aware imputation (same hour-of-day and day-of-week mean, not all-time mean).
- Keep EMA's hard NaN gate before training, but make it configurable (exclude Brent crude with 54% structural missingness from the check).
- Extend physical limits validation to SMARD data — prices within [-500, 1000] EUR/MWh, load must be positive, generation can't be negative (except pumped storage).
- Adopt EP's gap-aware cubic spline interpolation for both weather and market data.

### DST handling

The EPEX SPOT auction operates entirely in CET/CEST. A "delivery day" is midnight-to-midnight local German time. Spring-forward days have 23 delivery periods with no price for the skipped hour. Fall-back days have 25 delivery periods with two distinct prices for the repeated hour (typically labelled 3A and 3B).

EP's approach — convert to local time, interpolate the missing spring hour, average the duplicate autumn hour — correctly respects the market structure. This ensures a fixed 24-element target vector on every day.

Keep this approach but convert as late as possible: store everything in UTC through collection, storage, and feature engineering. Convert to local time only at target-creation (the daily pivot step). Document that spring-forward prices are interpolated (not real auction outcomes) and fall-back prices are averaged.

### Quarter-hourly regime

Post-October 2025, EPEX SPOT day-ahead prices are quarter-hourly (96 periods/day). For the initial merge, aggregate to hourly using the official EPEX 60-minute price index (arithmetic average of four 15-minute clearing prices). This preserves the existing model architecture. Native quarter-hourly forecasting is deferred to section 11.

---

## 4. Feature Engineering

### Current state

**EP** transforms hourly market data into a daily prediction problem using sklearn-compatible transformers:

- Derived market features: price spreads, net exports, generation mix percentages, price ratios.
- Temporal features: cyclical encoding (sin/cos), population-weighted multi-state holidays.
- Rolling window statistics via `RollingStatsTransformer` with `WindowSpec` dataclass.
- EWMA features with information-cutoff-aware computation.
- Same-hour lags (D-1, D-2, D-7, D-14).
- Daily pivot converting hourly → daily rows with 24-element target arrays.
- Leakage validation framework (`validate_pipeline_leakage()`) checking all transformers against availability rules.
- Pipeline versioning (v1–v5) for systematic ablation, defined in `pipelines.py` (1,750+ lines).

**EMA** engineers physics-informed features for weather-to-generation mapping:

- Per-target feature classes: `WeatherWindPowerFE` (air density, wind power density, wind shear, turbulence intensity, gust factor), `WeatherSolarPowerFE` (irradiance, clear-sky index, solar geometry), `WeatherLoadFE` (heating/cooling degree days).
- Spatial aggregation across multiple weather stations per TSO: mean, max, IDW, capacity-weighted. Method treated as a hyperparameter.
- Feature selection driven by Optuna — every toggle (compute air density? which lag windows?) is jointly optimised with model hyperparameters.

### Assessment

EP's leakage validation is excellent and has no equivalent in EMA. EMA's physics basis and spatial aggregation are the most valuable and hardest-to-reproduce parts of the project. The sklearn transformer pattern in EP adds ceremony without benefit for stateless transformations — nearly all transformers have trivial `fit()` methods that return `self`. EMA's code is procedural and monolithic (1,600-line `feature_eng.py`).

### Recommendations

**Architecture:** Three-layer design: self-explanatory config as orchestration → parsing function → feature engineering functions/classes.

**For EP's market features — a feature name DSL.** The feature name *is* the specification:
- `erzeugung_solar_d1` → average solar generation of D-1
- `target_price_h1` → target price at h-1
- `target_price_ewma_168_h12` → EWMA with span 168, cutoff at hour 12
- `forecast_wind_onshore_d0` → today's wind onshore forecast

A parser reads the name and dispatches to the right computation. The feature list is human-curated (~40-80 features) and serves as both specification and documentation. Leakage validation works at this level — the parser knows `_d1` means D-1 lag and checks against availability rules.

**For EMA's weather features — keep the config dict.** EMA's features are *computed from scratch* (should we compute air density at all? moist or dry? which spatial aggregation?), not transformations of existing columns. The config dict with boolean toggles and parameter values is the right shape. The config is also Optuna-searchable and needs to be serialisable.

**The two systems coexist cleanly.** EMA's config drives generation/load feature engineering and produces output columns (e.g., `forecast_wind_onshore`, `forecast_solar`, `forecast_load`). Those columns then become available to EP's feature name DSL like any other column.

**On classes vs functions:** Classes are justified only when stateful (learning parameters from data that must be reversed later). All of EP's current transformers except `FeatureScaler` and `TargetTransformer` have trivial `fit()` methods — replace them with plain functions. `compute_price_spreads(df)` is simpler than `PriceSpreadTransformer().fit(df).transform(df)`.

`FeatureScaler` and `TargetTransformer` belong with the model (see section 5), not the dataset preparation stage.

For EMA, keep the `WeatherBasedFE` class hierarchy — the config is Optuna-searchable, the spatial aggregation logic benefits from encapsulation, and the polymorphism is useful (looping over `[WindFE, SolarFE, LoadFE]`).

**Replace EP's pipeline versioning with composition.** Instead of v1–v5 as separate 200-line functions, use a config that selects which feature blocks to include:
```python
pipeline_config = {
    "base_features": True,
    "temporal": True,
    "rolling_target": {"windows": [7, 30], "stats": ["mean", "std"]},
    "ewma": {"spans": [6, 24, 168, 720, 2160], "cutoff_hour": 10},
    "same_hour_lags": {"offsets": [-1, -2, -7, -14]},
    "ema_forecasts": True,
    "neighbour_prices": True,
}
pipe = build_pipeline(pipeline_config)
```
Each "block" is a function returning pipeline steps. `build_pipeline` composes them.

**Port EP's leakage validation** to cover the combined pipeline, including EMA features. Weather forecasts are available days ahead (no lag), but weather actuals have reporting delays — the validation should catch accidental use of actual wind speed instead of forecast wind speed for day D.

---

## 5. Hyperparameter Tuning & Model Tracking

### Current state

**EP** uses MLflow with a local SQLite backend. Workflow: `build_pipeline()` caches a preprocessed dataset as an MLflow artifact (producing a `dataset_run_id`) → `train_and_log()` loads the dataset, trains, evaluates, logs metrics/params/model to MLflow. `tune_and_log()` accepts any sklearn search CV object. `train_final.py` retrains from committed hyperparameters in `blend_hyperparams.json`. Tuning was done in notebooks, not automated scripts.

**EMA** uses Optuna for joint feature+hyperparameter search with filesystem-based tracking (JSON files in a directory hierarchy). Each trial: propose feature config + model params → run full feature engineering → train with rolling CV → return RMSE. Results saved to `best_parameters.json`, `complete_study_results.csv`. No MLflow.

### Assessment

EP's dataset-run-id system creates clean lineage but adds workflow rigidity — you must build the pipeline and get a run_id before training, which makes rapid iteration awkward. EMA's joint optimisation is powerful for generation models but computationally infeasible for the combined price model (too many dimensions). EP's MLflow usage was messy in practice — overlapping experiments, inconsistent holdouts, no clear guidelines.

### MLflow usage guidelines

#### Experiment structure

**Experiments should map to what you're answering, not to model types.** Instead of experiments named "lgbm", "xgboost", "linear", use:

| Experiment | Purpose | What varies | What's held constant |
|---|---|---|---|
| `price/feature_selection` | Test different feature lists | Feature set | Model (e.g., LightGBM defaults), holdout, CV |
| `price/hyperparam_tuning/{model}` | Search model params | Hyperparameters | Feature set (best from above), holdout, CV |
| `price/blend_candidates` | Final trained models | Nothing — these are production | Everything fixed |
| `generation/{target}` | EMA per-target experiments | Features + model jointly | CV strategy |

**The key rule: within a single experiment, runs must be comparable.** Same holdout period, same CV strategy, same features (unless features are the thing being tested). If two runs used different holdout periods, they shouldn't be in the same experiment — this is what creates the overlapping mess.

#### Tags

Use tags to capture metadata for filtering:
```python
mlflow.set_tags({
    "stage": "feature_selection",       # or "hyperparam_tuning", "production"
    "feature_version": "v5",            # which feature list was used
    "holdout_days": "90",
    "cv_folds": "5",
    "cv_mode": "expanding",             # or "sliding"
    "target_transform": "log_shift",
})
```

#### Model naming

Use `{category}_{feature_version}` for model registry names (e.g., `lgbm_v5`, `xgboost_v5`). This groups related versions and makes the registry navigable.

#### When to create a new experiment vs a new run

- **New run:** changing model hyperparameters, trying a different model class, adjusting sample weights, changing target transform.
- **New experiment:** changing the feature set, changing the CV strategy, changing the holdout period, switching from daily to hourly architecture. These are structural changes that make runs non-comparable.

#### Run lifecycle

1. **Active** — current experiment, results being evaluated.
2. **Archived** — superseded by newer experiments, tagged `archived=true` with a reason. Still queryable but excluded from default views.
3. **Deleted** — clearly broken runs (crashed, wrong data, bugs). Delete rather than archive.

#### Helper functions

```python
def audit_experiment(experiment_name: str) -> pd.DataFrame:
    """Flag runs with missing tags, inconsistent holdout/CV/features.
    
    Checks: do all runs have the same holdout_days? Same cv_folds?
    Same feature_version (in non-feature-selection experiments)?
    Flags outliers for review.
    Returns DataFrame of flagged runs with reasons.
    """

def archive_runs(run_ids: list[str], reason: str = "superseded"):
    """Tag runs as archived with a reason. Excludes from default queries
    without deleting data. Reason is stored in tag 'archive_reason'."""

def get_best_run(experiment_name: str, metric: str = "mae", 
                 stage: str = None, exclude_archived: bool = True) -> dict:
    """Return the best run, optionally filtered by stage tag.
    Excludes archived runs by default."""

def compare_feature_sets(experiment_name: str) -> pd.DataFrame:
    """Return a DataFrame comparing all feature-set experiments.
    Columns: feature_version, n_features, mae, rmse, r2, n_runs.
    Only meaningful for feature_selection experiments."""

def compare_models(experiment_name: str, metric: str = "mae") -> pd.DataFrame:
    """Side-by-side comparison of model types within an experiment.
    Shows per-model-class best/mean/std of the target metric."""

def cleanup_orphaned_artifacts():
    """Find MLflow artifacts (models, datasets) not referenced by any
    active or archived run. List for manual review before deletion."""

def export_experiment_summary(experiment_name: str, path: str):
    """Export a self-contained summary of an experiment to markdown.
    Includes: purpose, best run, all runs table, tags, notes.
    Useful for documenting completed experiment rounds."""
```

The `audit_experiment` function is the most important. Run it before any selection step to catch inconsistencies.

#### Workflow for a typical experiment round

1. Create experiment with descriptive name.
2. Run experiments, tagging each run consistently.
3. Run `audit_experiment()` to check consistency.
4. Run `compare_feature_sets()` or `compare_models()` to pick the winner.
5. Export summary with `export_experiment_summary()`.
6. Archive superseded runs with `archive_runs()`.
7. Promote winner to the next stage.

### Tuning approach

- **For EMA generation models:** keep Optuna with joint feature+hyperparameter search (the search space is tractable and the interaction between features and model params is significant). Log trials to MLflow via `optuna.integration.MLflowCallback` for tracking.
- **For the price model:** grid search over model hyperparameters. The GBT hyperparameter landscape is smooth and the important parameters are well-known (learning rate, max depth, regularisation). A coarse grid over 3-4 parameters, then a finer grid around the best region, gets within a few percent of Optuna's optimum in a fraction of the time. Feature selection is human-curated via the feature name list.
- **After settling on features and hyperparams independently:** re-tune hyperparams on the final feature set to catch the most important interactions. This is one additional grid search, not a combinatorial explosion.

### Target transforms and scaling

`FeatureScaler` and `TargetTransformer` belong here, not in the dataset preparation stage. They are part of the sklearn-style model pipeline — `TransformedTargetRegressor` wraps the model with target scaling, and feature scaling is a pipeline step before the model. The choice of transform (log-shift, yeo-johnson, quantile, none) is a tunable parameter, evaluated alongside model hyperparameters.

### Conformal prediction intervals

Wrap each model in MAPIE's `MapieRegressor` during training:

```python
from mapie.regression import MapieRegressor

base_model = LGBMRegressor(**best_params)
model = MapieRegressor(base_model, method="plus", cv=5)
model.fit(X_train, y_train)
y_pred, y_intervals = model.predict(X_test, alpha=0.1)  # 90% interval
```

This adds minimal overhead to training and produces calibrated prediction intervals alongside point forecasts. Each model in the blend produces three outputs: point forecast, lower bound, upper bound.

EMA already uses MAPIE for generation forecasts. Port the same pattern to price models, noting EMA's fix for XGBoost float32 precision (`eps=1e-4` instead of MAPIE's default `1e-6`).

### Dataset caching

Simplify EP's `dataset_run_id` indirection. Instead: the feature name list + cleaning config deterministically produce a dataset. Hash the config to create a cache key. If the cache file exists and the source data hasn't changed, load it; otherwise rebuild. No MLflow run needed for the dataset itself.

### Time-series cross-validation

Adopt EMA's `compute_timeseries_split_cutoffs` for the price model. It ensures folds start/end at day boundaries, the train/test ratio is consistent, and the forecast horizon matches the actual 24-hour prediction task. EP's `TimeSeriesSplitter` is simpler but doesn't enforce these constraints, which matters when evaluating a day-ahead forecast as a coherent 24-hour block rather than individual hours.

---

## 6. Ensembling

### Current state

**EP** uses inverse-MAE weighted blending of 8 models (2 per category: linear, LightGBM, XGBoost, CatBoost). Selection pipeline: candidates from MLflow → CV validation → best-MAE + best-RMSE per category → train on all-minus-holdout → compute weights → save `blend_config.json`. Time-varying weights (rolling window) found to outperform stacking alternatives. Daily warm-starting for tree models. Biweekly full retrain.

**EMA** uses stacking ensembles — a meta-model (LightGBM or ElasticNet) trained on out-of-sample base model predictions. Optionally includes MAPIE prediction intervals as meta-features. Meta-model also Optuna-tuned.

### Recommendation

The time-varying inverse-MAE blend was found to outperform stacking alternatives through empirical testing. Continue with this approach; re-evaluate with the same notebook-based comparison when the combined repo's model set changes.

### Blending prediction intervals

With MAPIE-wrapped models (from section 5), each model produces a point forecast plus lower/upper bounds. Blend the intervals by weighted averaging:

```
blend_point = Σ(w_i × point_i)
blend_lower = Σ(w_i × lower_i)
blend_upper = Σ(w_i × upper_i)
```

This produces a blended prediction interval. The blend weights are the same inverse-MAE weights used for point forecasts.

At inference, the output JSON includes `forecast_lower` and `forecast_upper` alongside `forecast` for each hour, and the dashboard renders a shaded band around the point forecast.

---

## 7. Deployment

### Current state

**EP** runs two GitHub Actions workflows:
- **Daily forecast** (08:00 UTC): update data → EMA overlay (fetch CSV from GitHub Pages) → load 8 models → predict → blend → write JSON → deploy to GitHub Pages. Bootstrap via GitHub Releases. Feature snapshots saved for audit.
- **Biweekly retrain** (06:00 UTC, 1st/15th): update data → EMA overlay → retrain all models → recompute weights → commit `.joblib` files → upload merged dataset to release. Degradation check flags `needs_reselection`.

**EMA** runs a four-workflow chain: `collect_data` (07:00 UTC) → `update_forecasts` → `publish_forecasts` → `deploy`. Training done on-premises; only inference runs in CI.

### Recommendations

**Single daily workflow** for the combined repo:
```
collect weather data ─┐
collect market data  ─┤─→ run generation/load inference
                      │   → run price inference (using generation forecasts)
                      └─→ publish → deploy
```
Weather and market data collection can run in parallel (separate jobs with `needs` dependency on the subsequent inference step). Generation inference must complete before price inference. This replaces the two-repo, two-workflow-chain architecture and eliminates the fragile cross-repo bridge.

**Timing constraint:** The entire chain must complete by ~10:00 UTC to be useful before the 12:00 CET day-ahead auction. Profile current runtimes: EMA data collection is the bottleneck (~20 min for Open-Meteo across all locations). Consider parallelising weather data collection by asset type.

**Don't commit models or large data to git.** Use GitHub Releases for the merged dataset (EP already does this) and for trained models. The daily workflow downloads models from the release, runs inference, and uploads updated data. Retrain uploads new models to the release. This keeps the repo clean and avoids LFS growth.

**Unified dashboard** showing both price forecasts and generation/load forecasts in one place. Generation context explains price movements to users ("prices high tomorrow because wind generation forecast to drop"). Include the prediction interval band (from section 6) around the price forecast. Retain EP's existing monitoring tab, extended to cover generation/load forecast accuracy alongside price accuracy.

**Simplify inference code.** EP's 730-line `inference.py` handles data update, EMA overlay, pipeline loading, gap imputation, prediction, output formatting, snapshot management, error computation, and history management. Separate these into distinct functions with a clear orchestration layer. Data update should be a separate step (as EMA does).

**Local data sync.** Add `make sync` to pull the latest merged parquet from GitHub Releases into the local environment. This solves the stale-local-data problem when only CI updates data. Optionally, have `make update` check the release timestamp against local data and offer to download if the release is newer.

**Retrain workflow:** Update data → retrain all models (from committed hyperparameters) → recompute blend weights → check for degradation → upload new models to release → commit `blend_config.json`. The degradation check with `needs_reselection` (from EP) is a good production safeguard to keep.

---

## 8. Testing

Neither repo has meaningful test coverage. EP has no `tests/` directory. EMA has a `tests/` directory with minimal content.

### Recommended test structure

**Unit tests:**
- Physics functions: given known inputs, `compute_wind_power_density`, `compute_air_density`, etc. return expected values.
- Feature DSL parser: `parse_feature("target_price_ewma_168_h12")` returns the correct specification.
- Cleaning rules: each YAML rule type applied to a synthetic DataFrame produces expected output.
- Leakage validation: known-good and known-bad pipeline configurations pass/fail as expected.

**Integration tests:**
- Data pipeline round-trip: collect (from mocked API responses) → clean → feature engineer → verify expected shape and no NaN in required columns.
- Inference smoke test: full pipeline on a small data sample (last 30 days) runs without crashing and produces valid JSON output.
- Blend consistency: blend predictions are a weighted average of individual model predictions (exact equality within floating-point tolerance).

**Data quality checks (run in CI):**
- After each data update: verify no unexpected NaN in critical columns, verify timestamps are continuous, verify row counts are within expected range.
- After each retrain: verify blend MAE on holdout hasn't degraded beyond threshold.

The YAML cleaning rules are particularly testable — each rule is a pure input→output mapping, and synthetic DataFrames can exercise edge cases (what happens at the exact bidding area split timestamp? what if nuclear generation has NaN after decommissioning?).

---

## 9. Environment & Code Quality

**EP** uses conda + `pyproject.toml` with Python 3.13.5 and ruff for linting (`make lint` / `make format`).

**EMA** uses Pipenv with Python 3.11. No linting.

### Recommendations

- **Single environment** for the combined repo. Use `pyproject.toml` (the modern standard). Resolve dependency conflicts early — MAPIE 0.9.2 has API breaks with 1.x, and specific xgboost/catboost version constraints exist.
- **Standardise on ruff** with shared config. Run in CI as a pre-merge check.
- **Python version:** target 3.11 or 3.12 for maximum compatibility. 3.13.5 is bleeding-edge and may cause issues with some dependencies.

---

## 10. Implementation Order

The merge should follow the data flow, with each stage testable independently before moving to the next. The goal is to reproduce existing functionality from both repos in the new structure before attempting any new experiments.

### Phase 1: Foundation (weeks 1-2)

1. **Create new repo** with unified directory layout, `pyproject.toml`, ruff config. Get the environment working with all dependencies from both repos. This is likely the most frustrating step — dependency conflicts between MAPIE, xgboost, catboost, and the Python version will surface here.
2. **Port data collection.** Extend EP's `DataSource` to cover SMARD per-TSO regions and Open-Meteo. Extract `eu_locations.py` to JSON. Unify CLI. Verify all data sources download correctly and produce valid parquet.
3. **Port storage.** Parquet-from-the-start for all sources. Implement `save_parquet` utility with compression/dtype reduction. Verify incremental updates work for all sources.

**Milestone:** `make data` downloads all data from scratch and produces the correct directory structure with valid parquet files.

### Phase 2: Data Processing (weeks 3-4)

4. **Port cleaning.** Create the YAML cleaning rules config from EP's `handle_missing_values()`. Add EMA's physical limits validation. Implement the generic interpreter. Write unit tests for each rule type.
5. **Port feature engineering — EMA side.** Port physics functions as a standalone library of pure functions. Port `WeatherBasedFE` classes with Optuna integration. Verify generation/load forecasts reproduce EMA's current results.
6. **Port feature engineering — EP side.** Implement the feature name DSL and parser. Port the individual feature computation functions (rolling stats, EWMA, same-hour lags, daily pivot). Port leakage validation. Verify price features reproduce EP's current feature set.

**Milestone:** A feature list produces the same dataset as EP's current best pipeline, and EMA's generation forecasts match the current production output.

### Phase 3: Model Training (weeks 5-6)

7. **Set up MLflow** with the experiment structure from section 5. Implement helper functions (audit, archive, compare). Set up dataset caching with config hashing.
8. **Port model training.** Port `train_and_log`, `tune_and_log`, time-series CV. Add MAPIE wrapping for conformal intervals. Verify model training reproduces EP's current blend MAE within tolerance.
9. **Port ensembling.** Port blend selection, validation, weight computation. Add interval blending. Verify blend results match EP's current production.

**Milestone:** The full train → blend → evaluate pipeline runs and produces comparable metrics to the current EP production.

### Phase 4: Deployment (weeks 7-8)

10. **Port deployment.** Unified GitHub Actions workflow (collect → infer generation → infer price → deploy). Unified dashboard with price + generation forecasts + prediction intervals. `make sync` for local data.
11. **Testing.** Unit tests, integration tests, CI data quality checks.
12. **Documentation.** Update README, DATA.md, MLflow usage guide.

**Milestone:** Daily forecasts deploy automatically, the dashboard shows price + generation forecasts with prediction intervals, and the monitoring tab tracks accuracy for both.

---

## 11. Extensions

These are experiments and enhancements enabled by the unified architecture, to be tackled only after the merge is complete and existing functionality has been reproduced. Roughly ordered by expected value.

### Weather features directly in the price model

The highest-value experiment. Currently EMA forecasts generation from weather, then EP uses those forecasts. But the two-stage pipeline optimises for generation accuracy, not price-prediction accuracy. A single end-to-end model could learn which weather signals matter for *prices* directly — the timing of wind ramps relative to demand peaks might matter more for prices than total wind generation.

The unified feature DSL enables this: `wind_power_density_cap_weighted_d0` sits in the same feature list as `target_price_ewma_168_h12`. No architectural changes needed.

EMA's Optuna-driven feature selection is computationally infeasible for the full price model search space. Instead: fix physics features at sensible defaults (air density, wind power density, wind shear, cyclic direction), and let the model's feature importance identify what matters for prices. Spatial aggregation method (6-7 options) is worth including as a tunable.

### Per-TSO generation as price features

The geographic distribution of generation may contain price-relevant information beyond the national aggregate — e.g., wind concentrated in the north with congested transmission to the south. Architecturally free if both per-location and aggregated columns are available in the dataset. Test by adding per-TSO generation features to the feature list and comparing blend MAE against the national-aggregate-only baseline.

### Multi-day price forecasting

Extend from D+1 to D+2 through D+7 horizons. Key challenges:
- Leakage rules are horizon-dependent — for D+2, D-1 auction prices are unavailable, so most of EP's strongest features disappear.
- Options: recursive forecasting (predict D+1, use as input for D+2 — error accumulation risk) or direct multi-horizon (separate models/feature sets per horizon).
- The feature DSL would need horizon awareness: what `_d1` means depends on the forecast horizon.
- EMA's recursive framework could be adapted for prices, but price autocorrelation decays faster than generation autocorrelation.

### Quarter-hourly price model

Move from hourly-aggregated to native 15-minute forecasts. The 24-element target vector becomes a 96-element vector, or alternatively four separate models for each quarter-hour within each hour. Requires rethinking the daily pivot step and the leakage rules. Only worthwhile if there's a use case for quarter-hourly granularity (e.g., intraday trading, battery optimisation).

### API

Expose forecasts via a REST API rather than (or in addition to) the static GitHub Pages dashboard. Options:

- **Lightweight:** A simple Flask/FastAPI service that reads the deployed JSON files and serves them as endpoints. Could run on a free tier (Railway, Render, Fly.io). Endpoints: `GET /forecast/price/latest`, `GET /forecast/generation/latest`, `GET /forecast/price/history?days=30`. Minimal effort since the data is already produced by the daily workflow.
- **Full-featured:** An API that can generate forecasts on demand (not just serve pre-computed ones). Would need the model serving infrastructure — loading MAPIE-wrapped models, applying feature pipelines, running inference. Heavier to build and host but enables custom queries ("give me a forecast for a specific scenario").
- **Natural-language query interface:** A chatbot that converts plain-text questions into queries against the data. E.g., "what was the average price last week when wind generation was above 30GW?" This is where DuckDB becomes useful — text-to-SQL against parquet files. Could use an LLM (Claude API) for the natural-language-to-SQL translation.

For initial implementation, the lightweight option is the right starting point — it's a few hundred lines of code and adds genuine utility (programmatic access to forecasts for downstream consumers).

### DuckDB / database layer

Add a SQL query interface over the parquet files for analytical queries and multi-region expansion. DuckDB can query parquet directly with zero migration. Useful for: the natural-language query interface, complex analytical queries across time ranges, and as a foundation for a multi-region EU model where relational queries across bidding zones become natural.

### Multi-region EU model

Extend to forecast prices across multiple European bidding zones simultaneously. The data collection, storage, and cleaning architecture of the combined repo is designed to be region-parameterised (YAML cleaning rules per country, `DataSource` per region), so the extension is incremental rather than a rewrite. The main challenge is data acquisition — each bidding zone has its own SMARD-equivalent data source with different APIs and conventions.
