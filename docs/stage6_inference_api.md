# Stage 6: Inference, API & CI/CD

**Date written:** 2026-06-30  
**Preceded by:** Stage 5c complete (production ensemble holdout MAE 11.148)  
**Detailed plan for:** `docs/master_plan.md` §6

---

## Goal

Automated daily forecasts running end-to-end (data → gen/load inference → price inference → output) served via FastAPI, with GitHub Actions workflows for daily runs and periodic retraining.

---

## Key Challenges

**1. Gen/load inference feature construction.** At training, `_build_features` in `gen_load.py` calls the weather FE class with `weather_config` from the best Optuna trial. At inference, we must reconstruct the *same* weather FE with the *same* config, applied to current-forecast weather — not archive weather. The `best_config.json` artifact logged to MLflow already contains `weather_config`, but inference must load it without re-running Optuna.

**2. Upstream exog chaining at inference.** At training, `_load_upstream_actuals` provides ground-truth SMARD values for upstream targets (wind/solar as features for load, etc.). At inference this must be replaced by the model's own forecasts from the previous inference step. This asymmetry is intentional (mirrors EMA) but requires careful orchestration: wind/solar forecasts must be ready before load inference starts.

**3. Extending the merged dataset to the forecast date.** The price model needs features for *tomorrow* (D+1 delivery hours 0–23). The merged dataset ends at the most-recent SMARD observation. The extension must forward-fill known features (open commodity prices, SMARD day-ahead forecasts) and leave unavailable features at their D-1 values or NaN, consistent with the feature list's availability rules.

**4. Model storage.** All models currently live in MLflow (sqlite DB + artifact filesystem). For CI/CD, models need to be serializable to disk and downloadable at workflow time. A `gen_load_config.json` needs to record the best (target, region) run_id and config. The price ensemble is already in `models/ensemble_config.json` with run_ids.

---

## Architecture Overview

```
Daily pipeline (deploy/inference.py)
  ├── update_data()              → SMARD, weather, commodities incremental update
  ├── merge_and_process()        → re-run merge pipeline on updated raw data
  ├── run_gen_load_inference()   → per-target, per-region 168h forecasts
  │   ├── wind_onshore (×4 TSOs) ─┐
  │   ├── wind_offshore (×2 TSOs) ─┤ parallel, no dependencies
  │   ├── solar (×4 TSOs) ────────┘
  │   ├── load (×5 TSOs)          ← uses wind/solar outputs as exog
  │   └── gen_load_diff / DE_NATIONAL ← uses all above
  ├── aggregate_gen_load_national() → sum-of-regions per target
  ├── update_historical_forecasts() → write today's live forecast into hist parquet
  ├── run_price_inference()      → 24h price forecast using gen/load outputs
  ├── validate_outputs()         → sanity checks; fail before publishing bad data
  └── write_outputs()            → JSON/Parquet to deploy/data/
```

---

## Sub-stages

### 6.0 Prerequisites: gen_load_config.json and model serialization

**Status of what already exists:**
- `models/ensemble_config.json` — ✅ has price model run_ids, hyperparams, and ensemble weights
- `models/gen_load_config.json` — ❌ does not exist; must be generated

**What to build:**
- `deploy/model_store.py` — functions to export models from MLflow to disk and load them
- CLI extension: after `train gen-load` completes, write `models/gen_load_config.json` with the best (target, region) → {run_id, model_type, model_params, weather_config, dataset_params} for each combo
- `export-models` CLI command: reads `gen_load_config.json` and `ensemble_config.json`, downloads the corresponding MLflow artifacts, and serializes them to `models/gen_load/{target}_{region}.joblib` and `models/price/{run_id}.joblib` respectively
- Makefile: `export-models` target

**gen_load_config.json schema:**
```json
{
  "generated_at": "2026-06-30T12:00:00Z",
  "combos": {
    "wind_onshore/DE_50HZ": {
      "run_id": "abc123",
      "model_type": "XGBRegressor",
      "model_params": {...},
      "weather_config": {...},
      "dataset_params": {"log_target": false, "lags_target": 3, "scaler": "minmax"}
    },
    ...
  }
}
```

**Implementation notes:**
- `_pick_best_base_run` in `cli.py` already has the selection logic. After historical_forecasts export, call a new `_write_gen_load_config` helper that appends the chosen run_id + best_config.json to the JSON file.
- `model_store.py` uses `mlflow.MlflowClient().download_artifacts(run_id, "model")` to get the artifact path, then `joblib.dump(model, path)` to serialize. Price models from `train_model()` are stored under the `"model"` artifact path (standard `mlflow.sklearn.log_model` output).

### 6.1 Gen/Load Inference

**File:** `deploy/gen_load_inference.py`

**Function:** `run_gen_load_inference(forecast_date: date | None = None) -> dict[str, pd.DataFrame]`

**Steps per (target, region):**
1. Load `models/gen_load/{target}_{region}.joblib` (the fitted sklearn model+scaler from training)
2. Load gen_load_config for this combo: `weather_config`, `dataset_params`
3. Load current-forecast weather: `data/raw/weather/{weather_type}/{tso}/forecast.parquet`
4. Reconstruct weather FE: `fe_class(weather_config, locations)(df_forecast_weather)`
5. Compute temporal features from forecast timestamps
6. Populate exog features from *previous step's outputs* (not SMARD actuals)
   - For wind/solar: no exog needed
   - For load: use wind_onshore and wind_offshore forecast outputs from step above
   - For gen_load_diff: use wind_onshore, wind_offshore, solar, load outputs
7. Concatenate feature frames → X_forecast
8. Add target lag columns: last known actuals (from TSO parquet) for the lag seed, then recursive overwrite
9. Run `forecast_with_lags` (recursive, 168h) or `forecast_direct` depending on `lags_target`
10. Apply inverse log transform if `log_target=True`
11. Apply conformal PI calibration (from MLflow run metrics `conformal_quantile`)
12. Return DataFrame with `[y_pred, y_lower, y_upper]` indexed by forecast timestamps

**Dependency order (matches `GEN_LOAD_TRAINING_ORDER`):**
```python
# Wave 1 — can run in parallel (no exog dependencies)
wind_onshore_results, wind_offshore_results, solar_results = run_wave_1()

# Wave 2 — needs wind/solar forecasts
load_results = run_load_inference(
    exog_wind_on=wind_onshore_results,
    exog_wind_off=wind_offshore_results,
    exog_solar=solar_results,
)

# Wave 3 — needs all above
gen_load_diff_result = run_gen_load_diff_inference(
    exog_wind_on=wind_onshore_results,
    exog_wind_off=wind_offshore_results,
    exog_solar=solar_results,
    exog_load=load_results,
)
```

**Exog column naming at inference:** must match the TSO column names from `_load_upstream_actuals` (e.g. `wind_onshore_50hz` for the DE_50HZ region). The inference result DataFrames are keyed by region; building the exog frame replicates the loop in `_load_upstream_actuals`.

**Historical forecasts update:** After inference, append today's (y_pred, y_lower, y_upper) to `data/processed/historical_forecasts/{target}_{region}.parquet`. This keeps the file current so future price inference always finds the most recent forecasts. Use the forecast timestamps (not the run date) as the index.

### 6.2 Gen/Load National Aggregation

**File:** `deploy/gen_load_inference.py` (same file)

**Function:** `aggregate_national(results: dict) -> dict`

Sum per-TSO forecasts to national level, matching the `export_national_forecasts.py` pattern from EMA:
- For lower PI bounds: sum across TSOs (conservative — positive correlation)
- For upper PI bounds: same
- Writes to `data/processed/historical_forecasts/{target}_DE_NATIONAL.parquet` (live update)

### 6.3 Price Inference

**File:** `deploy/price_inference.py`

**Function:** `run_price_inference(forecast_date: date | None = None) -> pd.DataFrame`

**Steps:**
1. Load `models/ensemble_config.json`
2. Load `data/processed/merged.parquet`
3. Apply EMA overlay: call `_overlay_ema_forecasts` from `modeling/price.py` — this uses the freshly-updated `historical_forecasts/*.parquet` with today's live gen/load forecasts
4. Extend dataset to D+1 forecast horizon via `_extend_to_forecast_date(df)`:
   - Find the last row with `target_price` not NaN (the last known auction result, typically D-1)
   - Add 24 rows for D+1 delivery hours 0–23
   - Forward-fill: commodity prices (already daily), regime indicators, open market data
   - For SMARD day-ahead forecasts (`prognostizierte_*`): use the live values already fetched during `update_data` (SMARD publishes D+1 generation forecasts at ~18:00 CET D-1)
   - The existing `prognostizierte_*` columns for D+1 hours come from SMARD; the EMA overlay (step 3) replaces them with live gen/load model outputs for the D+1 period
5. Build price feature dataset: call `engineer_features(df_extended, feature_list)` for each unique `feature_version` in the ensemble config
6. For each base model in ensemble config:
   - Load `models/price/{run_id}.joblib`
   - Slice the appropriate feature version's dataset to D+1 rows
   - Apply scaler (if any) fitted on training data (stored as part of the pipeline)
   - `model.predict(X_d1)` → 24 scalar predictions
7. Apply ensemble weights (from `ensemble_config.json`): `y_blend = Σ(w_i × y_i)`
8. Apply conformal PI: `y_lower = y_blend - conformal_q`, `y_upper = y_blend + conformal_q`
9. Return DataFrame with 24 rows, columns `[forecast, forecast_lower, forecast_upper]`

**Dataset extension detail (`_extend_to_forecast_date`):**

The merged dataset is in tz-naive local time (Europe/Berlin delivery hours). Tomorrow's delivery hours are determined by:
```python
last_day = df.index[-1].date()
forecast_date = last_day + timedelta(days=1)
new_index = pd.date_range(
    f"{forecast_date} 00:00",
    f"{forecast_date} 23:00",
    freq="h",
)
```

Features that need special handling for D+1 rows:
- Commodity prices: forward-fill from last known row (daily data, known by midnight)
- `regime_*` indicators: compute from date
- `prognostizierte_*` columns: SMARD publishes these for D+1 around 18:00 CET. If already fetched (update_data ran after 18:00), they're in the raw data. If not, forward-fill from yesterday.
- All D-1 lag features (`price_d1`, `price_h24`, etc.): these reference yesterday's data which IS in the dataset, so rolling/shift operations on the extended df naturally populate them correctly.
- After the dataset extension, run `engineer_features` normally — the DSL handles lag computation.

**Note on scaler:** The price model pipeline is a `Pipeline([("scaler", scaler), ("model", TransformedTargetRegressor(...))])` (from `training.py`). When exported as a joblib, the scaler is part of the pipeline and will apply its learned statistics to the inference row. Linear models use RobustScaler; tree models use `none`. This is all handled transparently by the saved pipeline artifact.

### 6.4 Output Validation

**File:** `deploy/validation.py`

**Function:** `validate_outputs(price_df, gen_load_results) -> None`

Raises `ForecastValidationError` (fail fast, before any JSON is written) if:
- Price: any value outside [-500, 3000] EUR/MWh (EPEX bounds + buffer)
- Generation: any negative value
- Solar: any positive value during night hours (solar_elevation ≤ 0°)
- Load: any value below 1000 MW or above 120,000 MW (national)
- NaN in any forecast column
- Missing timestamps (not exactly 24 price rows, not exactly 168 gen/load rows)

### 6.5 Daily Pipeline Orchestration

**File:** `deploy/inference.py`

**Function:** `run_inference(skip_update: bool = False) -> dict`

Thin orchestrator:
```python
def run_inference(skip_update=False):
    if not skip_update:
        update_data()        # SMARD, weather, commodities
        merge_and_process()  # merge pipeline
    gen_load_results = run_gen_load_inference()
    aggregate_national(gen_load_results)
    update_historical_forecasts(gen_load_results)
    price_forecast = run_price_inference()
    validate_outputs(price_forecast, gen_load_results)
    write_outputs(price_forecast, gen_load_results)
    compute_errors()  # compare against actuals where available
    return {"price": price_forecast, "gen_load": gen_load_results}
```

**Error handling:** each major step is wrapped in try/except; partial failures logged and raised at the end so CI can distinguish data-update failures (might be transient) from model failures.

### 6.6 Output Writing

**File:** `deploy/publish.py`

**Function:** `write_outputs(price_forecast, gen_load_results, *, output_dir=DEPLOY_DATA_DIR)`

Output format aligned with `api/schemas.py` Pydantic models:

```
deploy/data/
├── price_forecast.json          # ForecastResponse for price
├── gen_load/
│   ├── wind_onshore_national.json
│   ├── wind_offshore_national.json
│   ├── solar_national.json
│   └── load_national.json
├── forecast_history.json        # ForecastHistoryResponse (rolling 30d)
├── model_metadata.json          # ModelsResponse
└── errors/
    └── {date}.json              # DailyError per day where actuals available
```

Also computes model errors against actuals (SMARD price data from the previous day) and writes to `errors/{yesterday}.json`.

### 6.7 FastAPI

**Files:** `api/app.py`, `api/routes.py`, `api/dependencies.py`

Port from EP's `src/api/` with extended endpoints for gen/load targets.

**app.py:**
```python
@asynccontextmanager
async def lifespan(app):
    app.state.models_loaded = deps.count_model_files()
    yield

app = FastAPI(title="Energy Forecasting API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])
app.include_router(router)
```

**Endpoints:**

| Method | Path | Description | Schema |
|--------|------|-------------|--------|
| GET | `/health` | API health | `HealthResponse` |
| GET | `/forecast/price` | Price forecast (24h, D+1) | `ForecastResponse` |
| GET | `/forecast/generation/{type}` | Gen forecast (168h, national) | `ForecastResponse` |
| GET | `/forecast/load` | Load forecast (168h, national) | `ForecastResponse` |
| GET | `/forecast/history` | Rolling 30d price history | `ForecastHistoryResponse` |
| GET | `/models` | Ensemble composition | `ModelsResponse` |
| GET | `/models/performance` | Daily errors | `PerformanceResponse` |

`{type}` path param: `wind_onshore`, `wind_offshore`, `solar`.

**dependencies.py** — thin wrappers that load from static JSON files in `deploy/data/`. No model loading; the API is stateless and serves pre-computed outputs.

### 6.8 Retrain Pipeline

**File:** `deploy/retrain.py`

**Function:** `run_retrain(holdout_days=None, force=False) -> dict`

**Price retrain:**
1. Load `ensemble_config.json` — identify the non-zero-weight models (5 in current production config)
2. For each base model: call `_finalize_gen_load_training` equivalent for price — specifically, call `train_model` with the stored hyperparams from the config (no Optuna re-search)
3. Re-run `compare_ensemble_methods` on fresh OOF → recompute weights via SLSQP
4. Check degradation: if `new_mae / old_mae > 1.20` → `needs_reselection = True`, skip model upload
5. If no degradation: update `models/ensemble_config.json`, export new models to `models/price/`

**Gen/load retrain:**
1. For each (target, region) in `gen_load_config.json`: call `retrain_gen_load_from_existing` (already implemented — reuses hyperparams, re-runs 218-fold CV)
2. Export new models
3. Update `gen_load_config.json` with new run_ids

**Degradation check:** compare 7-day rolling holdout MAE from the new run vs the stored metrics in config. Gen/load uses per-(target, region) checks separately.

**Note:** Full retrain of all non-zero-weight price models (5 models) takes ~30–60 min with VALIDATION_CV_FOLDS=5. Gen/load retrain (all 16 combos with 218 folds) takes ~8–12 hours — must run detached per `~/.claude/CLAUDE.md` § Long-Running Processes.

### 6.9 GitHub Actions

**File:** `.github/workflows/daily_forecast.yml`

```yaml
name: Daily Forecast
on:
  schedule:
    - cron: "0 8 * * *"  # 08:00 UTC = 09:00/10:00 CET, after D+1 SMARD forecasts publish
  workflow_dispatch:

jobs:
  collect-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up conda env
        # ... conda setup ...
      - name: Download models from Release
        run: gh release download latest -D models/ --clobber
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Update SMARD + weather + commodities (parallel)
        run: make update
      - name: Merge and process
        run: make process
      - name: Cache processed data
        uses: actions/upload-artifact@v4
        with:
          name: processed-data
          path: data/processed/

  inference:
    needs: collect-data
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: processed-data
          path: data/processed/
      - name: Download models from Release
        run: gh release download latest -D models/ --clobber
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Run inference
        run: energy-forecasting forecast --skip-update
      - name: Upload outputs
        uses: actions/upload-artifact@v4
        with:
          name: forecast-outputs
          path: deploy/data/

  deploy:
    needs: inference
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: forecast-outputs
          path: deploy/data/
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./deploy
```

**File:** `.github/workflows/retrain.yml`

```yaml
name: Retrain
on:
  schedule:
    - cron: "0 6 1,15 * *"   # 06:00 UTC, 1st and 15th of each month
  workflow_dispatch:
    inputs:
      force:
        description: "Force retrain even if no degradation"
        type: boolean
        default: false

jobs:
  retrain:
    runs-on: ubuntu-latest
    timeout-minutes: 120  # Price-only retrain; gen/load is manual
    steps:
      - uses: actions/checkout@v4
      - name: Download data from Release
        run: gh release download latest -D data/ --clobber
      - name: Update data
        run: make update
      - name: Run price retrain
        run: energy-forecasting retrain --price-only
      - name: Upload new models to Release
        if: success()
        run: gh release upload latest models/price/*.joblib models/ensemble_config.json --clobber
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Commit updated config
        if: success()
        run: |
          git config user.email "github-actions@github.com"
          git config user.name "GitHub Actions"
          git add models/ensemble_config.json
          git diff --cached --quiet || git commit -m "ci: update ensemble_config after retrain"
          git push
```

**Note on gen/load retrain:** At 8–12 hours, gen/load retrain doesn't fit in a standard GitHub Actions timeout. This is a known limitation. Options:
1. Run manually on the tower (`make retrain-gen-load` detached), then `make export-models` and upload to Release
2. Self-hosted runner on the tower (future improvement)
For Stage 6, gen/load retrain stays manual; only price retrain is automated.

### 6.10 CLI & Makefile Extensions

**New CLI commands:**

```
energy-forecasting forecast            # Run full daily inference pipeline
energy-forecasting forecast --skip-update  # Skip data update step
energy-forecasting retrain             # Full price retrain
energy-forecasting retrain --price-only    # Only retrain price models
energy-forecasting serve               # Start FastAPI server
energy-forecasting export-models       # Export MLflow artifacts to disk
```

**New Makefile targets:**
```makefile
forecast:      # make forecast
serve:         # make serve  
retrain:       # make retrain
export-models: # make export-models
sync:          # pull latest data/models from GitHub Release
```

### 6.11 Testing Strategy

**Unit tests:**
- `test_gen_load_inference.py`: mock weather + TSO data, verify feature construction, exog chaining, recursive forecast
- `test_price_inference.py`: mock merged data extension, verify 24h forecast output shape
- `test_validation.py`: verify each sanity check catches known-bad outputs
- `test_publish.py`: verify JSON schema matches Pydantic models
- `test_api.py`: TestClient on all endpoints with mock static data files

**Integration tests (require data):**
- `test_inference_smoke.py`: full `run_inference(skip_update=True)` on last 30 days; verify valid JSON output, no NaN, correct timestamp ranges
- `test_blend_consistency.py`: price ensemble output = Σ(w_i × model_i_output) within float tolerance

---

## Implementation Order & Dependencies

```
6.0 model_store.py + gen_load_config.json ← prerequisite for all inference
    ↓
6.1 gen_load_inference.py  ←── weather FE reconstruction
    ↓
6.2 aggregate_national (inside 6.1 file)
    ↓
6.3 price_inference.py   ←── depends on 6.1 (historical_forecasts updated)
    ↓
6.4 validation.py        (independent)
6.5 inference.py         ←── orchestrates 6.1–6.4
6.6 publish.py           ←── called by 6.5
    ↓
6.7 FastAPI (app, routes, deps)   ←── reads publish.py outputs
6.8 retrain.py           (independent; uses 6.0 model_store)
6.9 GitHub Actions       ←── wire everything together
6.10 CLI + Makefile      ←── expose all above
```

---

## Open Questions to Resolve During Implementation

**OQ1: Scaler serialization in gen/load models.** The gen/load training uses a bare `_ScaledLogPredictor` wrapper (not a sklearn Pipeline) for the recursive forecast path. This wrapper holds a `scaler` (from `_fit_scaler`) and a `log_target` flag. When exported to joblib, does this serialize correctly? Verify with a smoke test: `joblib.dump(predictor, path); loaded = joblib.load(path); loaded.predict(X_test)`.

**OQ2: Weather data availability at inference time.** `forecast.parquet` (Open-Meteo "current forecast" endpoint) provides 14-day ahead forecasts and is updated at each data collection run. At 08:00 UTC, D+1 (tomorrow) through D+14 forecast weather is available. Verify the `forecast.parquet` timestamp range includes D+1 hours 0–23 UTC (=D+1 hours 1–24 CET during winter, 2–25 CET during summer, adjusted for delivery hours).

**OQ3: Price feature extension with D+1 SMARD forecasts.** SMARD publishes day-ahead generation forecasts (`prognostizierte_*`) for D+1 around 18:00 CET D-1. The daily workflow runs at 08:00 UTC = 09:00/10:00 CET D, so these forecasts were published ~15 hours earlier and are already in the raw SMARD data. Confirm that `SmardSource.update()` actually fetches these forecast values (as opposed to only fetching settled actual data).

**OQ4: What to do about gen/load PI at inference time.** The gen/load models that use recursive lags bypass MAPIE (as documented in 5b). Their `y_lower/y_upper` columns are NaN in historical_forecasts parquets. At inference, we apply the `conformal_quantile` from the ensemble run (if the best run was an ensemble) or from the base model's run metrics. If the chosen model is a recursive-lag model with no conformal calibration, report PI as None in the API response.

---

## Milestone

- `make forecast` runs end-to-end locally without errors and produces `deploy/data/price_forecast.json`
- `make serve` starts FastAPI and all 7 endpoints return valid responses
- `make forecast --skip-update` runs in under 5 minutes (model loading + inference)
- Daily GitHub Actions workflow runs end-to-end (tested via `workflow_dispatch`)
- `make retrain` (price only) completes in under 90 minutes and updates `ensemble_config.json`
- Price forecast accuracy matches the 5c production run within 5% MAE on a manual spot check
- All new unit tests passing; API tests passing

---

## Stage 6 Evaluation

<!-- Fill in after Stage 6 is complete -->

**Status:** Not started

**What was implemented:**

**Deviations from plan:**

**Challenges encountered:**

**Insights for later stages:**
