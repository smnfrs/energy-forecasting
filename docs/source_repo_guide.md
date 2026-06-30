# Source Repo Guide

Quick reference for finding things in the two precursor repos. Designed for a Claude instance performing the merge.

**Repos:**
- `~/projects/energy_prices/` (EP) — price forecasting
- `~/projects/energy_market_analysis/` (EMA) — generation/load forecasting

**Companion docs:**
- `docs/merge_evaluation.md` — detailed comparison of both repos
- `docs/merge_decisions.md` — 17 architectural decisions for the merged repo
- `docs/master_plan.md` — implementation plan
- `EP_EMA_merge_plan.md` — an earlier merge plan (partially superseded by merge_decisions.md)

---

## energy_prices (EP)

**Location:** `~/projects/energy_prices/`
**Python:** 3.13.5 (strictly pinned)
**Package manager:** pip + pyproject.toml
**Entry point:** `src/cli.py` (Typer CLI)

### Directory Layout

```
src/
├── cli.py              # All CLI commands: download, update, combine, merge, blend, forecast
├── config/             # All configuration (path constants, column lists, pipeline definitions)
├── data/               # Data acquisition and processing
├── features/           # Feature engineering (sklearn transformers — being replaced)
├── modeling/           # Training, blending, baselines, metrics
├── deploy/             # Inference and retrain scripts
└── api/                # FastAPI REST layer
```

### Where to Find Things

#### Data Collection
| What | Where |
|------|-------|
| DataSource base class + all sources | `src/data/sources.py` — abstract base with `update()`, then SmardSource, IcapSource, YahooSource, FredSource, EnergyChartsSource |
| SMARD API functions (get_timestamps, get_data) | `src/data/smard.py` |
| Commodity download + gap reconstruction | `src/data/commodities.py` — TTF reconstruction (~line 548), carbon dual-source (~line 682), bias correction math throughout |
| SMARD key→name mappings (44 filters, cross-border flow codes) | `src/config/smard.py` |
| Commodity ticker mappings, date constants | `src/config/commodities.py` |
| EMA overlay (fetching gen/load forecasts from the other repo) | `src/data/ema.py` |

#### Data Processing
| What | Where |
|------|-------|
| CSV→Parquet combination (wide pivot) | `src/data/processing.py` — `combine_data()` (~line 117), `combine_data_incremental()` (~line 140) |
| DE-AT-LU + DE-LU merge, regime indicators | `src/data/processing.py` — `run_merge_pipeline()` (~line 483), `run_merge_pipeline_incremental()` (~line 548) |
| Unified target_price creation | `src/data/processing.py` — `create_unified_target()` (~line 367) |
| Missing value handling rules | `src/config/processing.py` — column lists and regime dates |
| Missing value implementation | `src/features/transforms.py` — `handle_missing_values()` (first ~300 lines) |

#### Feature Engineering

**WARNING:** The sklearn pipeline system (TransformerMixin, v1-v5 versioning) is being replaced by a suffix DSL. Do not replicate the sklearn patterns. Extract the underlying computation logic only.

| What | Where | Notes |
|------|-------|-------|
| **Current production pipeline** | `src/config/pipelines.py` — `preprocessor_v5_slim_hourly()` (~line 1200) | v1-v4 above it are dead code |
| Feature transformer classes | `src/features/transforms.py` — PriceSpreadTransformer, NetExportTransformer, GenerationPercentageTransformer, TemporalFeatureTransformer, GermanHolidayTransformer, etc. | Extract the math from `transform()` methods, ignore the sklearn ceremony |
| Time-series transformers (rolling, EWMA, lags) | `src/features/ts_transforms.py` — RollingStatsTransformer, EWMATransformer, SameHourLagTransformer, DailyPivotTransformer | Key algorithms worth porting |
| Rolling window specifications | `src/config/features.py` — ROLLING_FEATURE_SPECS_v5 (~line 200) |
| Leakage/availability rules | `src/config/features.py` — AVAILABILITY_RULES (51 rules, ~line 350) |
| Leakage validation implementation | `src/features/validation.py` — `validate_pipeline_leakage()` |
| Feature column definitions (neighbours, flow pairs, generation sources) | `src/config/features.py` — NEIGHBOR_PRICES, FLOW_PAIRS, GENERATION_COLUMNS (first ~100 lines) |
| Temporal config (German state populations for holidays) | `src/config/temporal.py` |
| Dataset caching in MLflow | `src/features/preprocessors.py` — `build_pipeline()`, `load_dataset()` |

#### Model Training
| What | Where |
|------|-------|
| Main training function | `src/modeling/training.py` — `train_and_log()` (~line 41) |
| Sample weighting (exponential decay) | `src/modeling/training.py` — `compute_sample_weights()` (~line 391) |
| Time-series CV splitter | `src/modeling/training.py` — `TimeSeriesSplitter` class (~line 431) |
| Blend ensemble (full pipeline) | `src/modeling/blend.py` — select_candidates, validate_candidates, select_final_models, train_and_blend, retrain_blend |
| Blend config structure | `models/production/blend_config.json` |
| Committed hyperparameters | `models/production/blend_hyperparams.json` |
| Retrain from committed params | `src/modeling/train_final.py` |
| Baseline models | `src/modeling/baselines.py` |
| Metrics | `src/modeling/metrics.py` — `calculate_metrics()` |
| Blend constants (holdout days, degradation threshold, etc.) | `src/config/modeling.py` |

#### Deployment
| What | Where |
|------|-------|
| Daily inference pipeline | `src/deploy/inference.py` — `run_inference()` (730 lines, needs splitting) |
| Retrain pipeline | `src/deploy/retrain.py` — `run_retrain()` |
| JSON output format | `src/deploy/inference.py` — `_write_output()` (~line 624) |
| Daily workflow | `.github/workflows/daily_forecast.yml` |
| Retrain workflow | `.github/workflows/retrain.yml` |

#### API
| What | Where |
|------|-------|
| FastAPI app + CORS setup | `src/api/app.py` |
| 5 endpoints (health, forecast, history, models, performance) | `src/api/routes.py` |
| Response schemas | `src/api/schemas.py` |
| Data loading helpers | `src/api/dependencies.py` |

#### Tests

**IMPORTANT:** Tests are gitignored. They exist locally but won't appear on GitHub. 11 test files in `tests/`:
- `test_data.py`, `test_daily_transforms.py`, `test_hourly_transforms.py`, `test_api.py`, `test_leakage.py`, `test_baselines.py`, `test_metrics.py`, `test_preprocessors.py`, `test_training.py`, `test_temporal_features.py`, `test_lag_transforms.py`

#### Other
| What | Where |
|------|-------|
| All Makefile targets | `Makefile` — data, update, lint, format, train-baselines, blend, forecast, api, etc. |
| Data documentation | `DATA.md` — comprehensive explanation of every column and data source |
| Developer guidelines | `CLAUDE.md` |
| MLflow database | `mlflow.db` (SQLite) |
| Production models | `models/production/*.joblib` |

### Gotchas

1. **Tests are gitignored.** You must work locally to access them.
2. **pipelines.py is 1759 lines.** Only v5 slim (~line 1200) is production. Everything above is dead code.
3. **inference.py is 730 lines.** It handles data update, EMA overlay, extension to forecast date, prediction, and output writing in one monolith. The sub-functions are `_update_data`, `_apply_ema_overlay`, `_extend_to_forecast_date`, `_predict_all_models`, `_write_output`.
4. **Datasets tracked as models in MLflow.** This was a mistake — the model registry was used for datasets. Don't replicate this pattern.
5. **Energy Charts source** is a secondary price data source used as a fallback when SMARD has gaps. Not critical.
6. **The EMA overlay** (`src/data/ema.py`) fetches forecasts from the *other* repo's GitHub Pages. This bridge is eliminated in the merged repo.

---

## energy_market_analysis (EMA)

**Location:** `~/projects/energy_market_analysis/`
**Python:** 3.11.5
**Package manager:** pipenv (Pipfile + Pipfile.lock)
**Entry points:** 4 separate CLI scripts (not a unified CLI)

### Directory Layout

```
data_collection_modules/    # Data acquisition (separate from data processing)
data_modules/               # Feature engineering and data preprocessing
forecasting_modules/        # ML pipeline, models, ensembles
database/                   # Raw collected data (Parquet, committed to git)
output/                     # Trained models and forecasts
deploy/                     # Static website + published data
tests/                      # Minimal (2 files)
```

**Note:** This is a flat layout, not a Python package. Files are imported via `from data_collection_modules.collect_data_smard_v2 import ...` etc. There's no `src/` or `__init__.py` at the top level.

### Where to Find Things

#### Data Collection
| What | Where |
|------|-------|
| Per-TSO SMARD collection | `data_collection_modules/collect_data_smard_v2.py` — `update_smard_v2()` (~line 160). TSO mappings at top (~line 23). Known-missing combinations (~line 50). |
| Open-Meteo weather collection (3 endpoints) | `data_collection_modules/collect_data_openmeteo.py` — `OpenMeteo` class. Hourly variables (~line 24), 15-min variables (~line 52), physical limits (~line 78). Three collection methods: `_collect_past_actual()`, `_collect_past_forecast()`, `_collect_forecast()`. |
| National SMARD (legacy) | `data_collection_modules/collect_data_smard.py` — `DataEnergySMARD` class |
| EPEX SPOT prices | `data_collection_modules/collect_data_epexspot.py` |
| **Location database (30+ locations)** | `data_collection_modules/eu_locations.py` — ~1800 lines of Python dicts. Cities (~line 24), offshore wind farms, onshore wind farms, solar farms. Per-TSO with lat/lon, capacity, roughness length. `countries_metadata` at bottom (~line 1769). |
| Parquet I/O with compression | `data_collection_modules/parquet_operations.py` — `ParquetOperations` class (zstd level 4, dtype reduction) |
| Validation utilities | `data_collection_modules/utils.py` |
| **CLI orchestrator** | `update_database.py` — `main_country()` (~line 26). Dispatches to 8 collection tasks. |

#### Data Processing
| What | Where |
|------|-------|
| Database extraction + merging | `data_modules/data_loaders.py` — `extract_from_database()` (~line 107). This is the main ETL: loads weather + SMARD, merges by TSO, selects features by target. |
| SMARD NaN imputation (column mean) | `data_modules/data_loaders.py` — `impute_smard_nans()` (~line 31) |
| Derived targets (gen_load_diff, residual_load) | `data_modules/data_loaders.py` — `compute_gen_load_diff()` (~line 44), `compute_residual_load()` (~line 20) |
| Weather data merging across TSOs | `data_modules/utils.py` — `merge_tso_dataframes()` (~line 102). Finds common date range, inner-joins. |
| NaN interpolation (linear, max gap 48h) | `data_modules/utils.py` — `handle_nans_with_interpolation()` (~line 9) |
| Periodicity enforcement | `data_modules/utils.py` — `fix_broken_periodicity_with_interpolation()` (~line 44) |

#### Feature Engineering
| What | Where |
|------|-------|
| **Physics formulas (all of them)** | `data_modules/feature_eng.py` — air density (~line 74), dew point (~line 124), wind shear (~line 153), turbulence (~line 171), wind power density (~line 210), wind chill (~line 191), humidex (~line 201), vapour pressure (~line 143) |
| **Spatial aggregation** | `data_modules/feature_eng.py` — `_haversine_distance()` (~line 218), `_weighted_average()` (~line 240), aggregation methods in each FE class |
| **WeatherWindPowerFE** | `data_modules/feature_eng.py` — class starts ~line 340. Options list ~line 344. `_preprocess_location()` ~line 366. Optuna suggestions ~line 596. |
| **WeatherSolarPowerFE** | `data_modules/feature_eng.py` — class starts ~line 667. Radiation features ~line 742. Optuna ~line 940. |
| **WeatherLoadFE** | `data_modules/feature_eng.py` — class starts ~line 1085. HDD/CDD ~line 1182. Optuna ~line 1410. |
| FE integration (dispatch to correct class) | `data_modules/feature_eng.py` — `physics_informed_feature_engineering()` (~line 1566) |
| Temporal features (cyclical, holidays) | `data_modules/feature_eng.py` — `create_time_features()` (~line 44) |
| Dataset class (scaling, imputation, train/test) | `data_modules/data_classes.py` — `HistForecastDataset` class (~line 131). `process_data()` (~line 243) is the main method. |
| Optuna suggestions for dataset params | `data_modules/data_classes.py` — `suggest_values_for_ds_pars_optuna()` (~line 102) |

#### Model Training
| What | Where |
|------|-------|
| **Pipeline orchestrator** | `forecasting_modules/interface.py` — `main_forecasting_pipeline()`. Routes to finetune/train/forecast/evaluate. |
| **Single-target models (MAPIE-wrapped)** | `forecasting_modules/base_models.py` — `BaseForecaster`, `XGBoostMapieRegressor`, `LGBMMapieRegressor`, `ElasticNetMapieRegressor`. MAPIE setup ~line 613. |
| **Recursive multi-step forecasting** | `forecasting_modules/base_models.py` — `forecast_window()` (~line 168). This is the key algorithm: predict one step, feed prediction back as lag, repeat 168 times. |
| **Multi-target models** | `forecasting_modules/base_models_multitarget.py` — `MultiTargetCatBoost`, `MultiTargetLGBM` |
| **Task executor** (finetune/train/forecast) | `forecasting_modules/tasks.py` — `ForecastingTaskSingleTarget`, `EnsembleModelTasks`. Finetune method ~line 477. Train ~line 592. Forecast ~line 622. Ensemble forecast ~line 889. |
| **Hyperparameter search spaces** | `forecasting_modules/hyperparameters_for_optuna.py` — `get_parameters_for_optuna_trial()`. LightGBM, XGBoost, CatBoost, ElasticNet, Prophet spaces. |
| **Time-series CV splitter** | `forecasting_modules/utils.py` — `compute_timeseries_split_cutoffs()` (~line 42). Day-boundary-aware. |
| **Metrics** | `forecasting_modules/model_evaluator_utils.py` — `compute_error_metrics()`. RMSE, MAE, sMAPE, PI coverage, PI width. |
| Optuna result saving | `forecasting_modules/utils.py` |

#### Configuration

**IMPORTANT:** EMA has no separate config module. Configuration lives inside the CLI scripts.

| What | Where |
|------|-------|
| **All task definitions** (targets, models, hyperparams, dataset params) | `update_forecasts.py` — lines 36-200. This is where targets, model types, finetuning trials, ensemble definitions, and dataset parameters are all configured. |
| **Ensemble definitions** | `update_forecasts.py` — look for `'ensemble[LightGBM](LightGBM,ElasticNet)'` strings (~line 106) |
| CLI argument parsing | `update_forecasts.py` — `adjust_and_run_for_tasklist()` (~line 325) |

#### Deployment
| What | Where |
|------|-------|
| National aggregation (per-TSO → DE/LU) | `export_national_forecasts.py` — CI propagation (sum lower/upper bounds for conservative intervals) |
| JSON/CSV export for dashboard | `publish_data.py` |
| Historical forecast generation (backtesting) | `generate_historical_forecasts.py` |
| Dashboard HTML | `deploy/index.html` |
| Dashboard JS (ApexCharts, 37KB) | `deploy/script.js` |
| Multi-language translations | `deploy/translations.json` |
| CSS (Bootstrap 5) | `deploy/assets/css/styles.css` |

#### CI/CD
| What | Where |
|------|-------|
| Data collection workflow | `.github/workflows/collect_data.yml` — 7:00 UTC daily |
| Forecast workflow | `.github/workflows/update_forecasts.yml` — triggered after collect_data |
| Publish workflow | `.github/workflows/publish_forecasts.yml` — triggered after update_forecasts |
| Deploy workflow | `.github/workflows/deploy.yml` — triggered after publish_forecasts |

The 4 workflows chain via `workflow_run` triggers. Each commits to main and pushes.

#### Tests

Minimal: `tests/test_data_loaders.py` (imputation, gen_load_diff, residual_load) and `tests/test_export_national.py`.

### Gotchas

1. **Config is in the CLI script.** All task definitions (which targets, which models, which hyperparams) live in `update_forecasts.py` lines 36-200, not in a config file.
2. **Database committed to git.** `database/` contains Parquet files that are committed directly. This causes repo bloat.
3. **No Python package structure.** No `src/`, no top-level `__init__.py`. Imports are `from data_collection_modules.X import Y`.
4. **MAPIE eps bug fix.** In `base_models.py` ~line 613, the conformity score eps is set to `1e-4` instead of the default `1e-6` because XGBoost 3.x returns float32 metrics that fall below float32 precision at `1e-6`. This fix must be carried forward.
5. **`best_model.json` files.** Each target directory has a `best_model.json` that records which model (including ensemble) performed best. The forecast pipeline reads this to know which model to use for each target.
6. **Exogenous variable loading.** `data_loaders.py` `extract_from_database()` loads forecasts from *other targets' best models* as exogenous features. For example, the load model uses wind and solar forecasts as inputs. This creates a dependency order during inference.
7. **Three Open-Meteo endpoints.** Archive (actuals, 2015+), historical forecast (forecasts as issued, ~2022+), current forecast (14-day). The distinction between actual and forecast weather is important for training — training on actual weather overfits relative to inference conditions.
8. **NumpyEncoder in tasks.py.** Custom JSON encoder for numpy types — XGBoost 3.x returns np.float32 metrics that standard json.JSONEncoder can't handle.
9. **Solar elevation computed at collection time.** `add_solar_elevation_and_azimuth()` in `collect_data_openmeteo.py` runs pysolar during data collection, not feature engineering. The computed columns live in the weather Parquet files.
10. **Stacking ensemble OOF.** The meta-model is trained on out-of-fold predictions from base models. The `cv_folds_base` parameter controls how many folds generate OOF data. Must be high enough for good coverage but not so high that it's slow.
