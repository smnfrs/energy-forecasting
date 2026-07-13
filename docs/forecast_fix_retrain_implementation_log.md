# Forecast Fix Retrain Implementation Log

**Started:** 2026-07-13
**Plan:** `docs/forecast_fix_retrain_plan.md`

This log records implementation checkpoints, decisions, and discoveries made while executing the retrain plan.

## 2026-07-13 — Workflow Disable Checkpoint

Commit `0bbcdbc` disables the daily price step during the retrain window. Instead of only changing GitHub Actions YAML, the implementation adds a testable `energy-forecasting deploy forecast --no-price` path:

- Gen/load inference and historical forecast updates still run.
- Gen/load dashboard outputs, actuals, gen/load errors, hindcast, and error summaries are refreshed.
- Existing `price_forecast.json`, `forecast_history.json`, price SHAP, price feature audit, and price model metadata are left untouched.
- The forecast narrative workflow step is explicitly skipped while price is disabled because it depends on price SHAP/facts.

Validation: `conda run -n energy-forecasting pytest tests/test_deploy_inference.py tests/test_cli.py -q` passed (10 tests).

## 2026-07-13 — Phase 0 Cleanup and Feature Contract

Implementation notes for the second checkpoint:

- Removed the dead legacy `load_gen_load_forecasts` API and helper constants from `modeling/gen_load_forecasts.py`; kept `_align_tz` because `build_forecast_columns` still needs it for UTC artifact alignment.
- Removed tests that only exercised the deleted `_derived_forecast_*` loader. Production behavior remains covered by `tests/test_forecast_inputs.py`.
- Kept `_normalize_local_delivery_grid` in `forecast_inputs.py`, but documented it as a defensive path for already-local tz-naive manual/test artifacts. The production UTC artifact path still uses `normalize_dst`.
- Added `FEATURE_CONTRACT = "forecast_v1"` and made `feature_contract` a required `TrackedRun` tag.
- Passed `feature_contract=forecast_v1` through feature selection, tuning, final price retraining, price production bakeoff, gen/load training/ensembles, and deploy retrain.
- Updated `docs/mlflow_conventions.md` and added `tests/test_mlflow_utils.py` coverage for missing/blank `feature_contract`.

Discovery: `SHORT_NAMES` already retains `prog_*` aliases only for backward-compatible parsing/audit while production price validation rejects `prog_`/`pct_prog_` tokens, matching the plan's intended boundary.

Validation: focused suite passed: `conda run -n energy-forecasting pytest tests/test_mlflow_utils.py tests/test_price.py tests/test_5c_derivations.py tests/test_forecast_inputs.py tests/test_deploy_inference.py tests/test_cli.py -q` (40 tests), plus `python -m py_compile` over touched modules.

## 2026-07-13 — Preservation and MLflow Archive

Preservation exports were written to `docs/archive/price_pre_forecast_contract/` before dataset regeneration or retraining:

- `price_feature_selection`: 71 runs exported and archived.
- `price_model_training`: 6,167 runs exported and archived. This is higher than the 5,000-run count in the plan; the live MLflow store had additional price tuning runs.
- `price_production`: 5 runs exported and archived.
- Current `models/ensemble_config.json`, `models/price_feature_cols.json`, production hyperparameters, ensemble weights, conformal settings, and the leakage-inflated 11.148 MAE baseline were copied into the archive.

Decision/change during implementation: the initial MLflow API archive pass was interrupted because per-run `set_tag` calls were too slow for 6,243 runs. The final archive operation uses a single idempotent SQLite transaction that sets `archived=true`, `archive_reason=pre-forecast-contract; leaky/non-comparable`, and `feature_contract=prog_leaky` for experiments `price/feature_selection`, `price/model_training`, and `price/production`. Verification query showed all 6,243 price runs have both archive and `prog_leaky` tags. No artifacts were deleted.

## 2026-07-13 — Dataset Regeneration and Audit

Stale cached price datasets were removed with `rm -f data/processed/datasets/price_*.parquet`, then clean base datasets were regenerated:

- `price_slim.parquet`: 15,535,914 bytes
- `price_full.parquet`: 21,439,053 bytes
- `price_max.parquet`: 39,302,664 bytes

The combined regeneration command was interrupted after `slim` and `full` completed because `conda run` buffered output and gave no visibility for several minutes. `max` was rerun separately and completed successfully; the warnings were pandas fragmentation warnings from iterative feature insertion, not correctness failures.

`python scripts/forecast_fix_dataset_audit.py` passed and wrote `dataset_audit.{json,md}`, `forecast_source_counts_by_year.csv`, and `forecast_gen_total_boundary.png` under `docs/archive/price_pre_forecast_contract/`:

- 100,824 merged rows.
- Forecast source mix: own=36,842 / smard=63,574 / actual=408 / missing=0.
- All eight `forecast_*` columns have 0 NaN.
- First own forecast timestamp: 2022-01-15 01:00.
- Own 2022+ residual identity max absolute error: 0.0.
- 2022+ `forecast_residual_load` differs materially from old `prog_residual`: mean absolute diff 2,918 MW, max diff 37,436 MW, 36,833 rows differ by more than 1 MW.
- Holdout window source mix: own=54 / smard=2,106 / actual=0 / missing=0.
- Regenerated dataset schemas contain no `prog_`, `pct_prog_`, or `prognostiziert` tokens.

Decision/change during implementation: the first residual identity gate incorrectly checked all fallback rows, including SMARD rows where operator residual is not required to equal the derived wind/PV sum. The gate was corrected to the plan's intended scope: 2022+ rows sourced from own forecast artifacts.

## 2026-07-13 — Optuna Study Isolation Guard

Before launching Stage A/B, existing `data/optuna` contents showed pre-contract study DBs with names like `fs_rfecv_optimum__LGBMRegressor__stage2_grid.db`. Because the tuning code used only `{feature_version}__{model_type}__{step}` as the study name with `load_if_exists=True`, a retrain that rediscovered `fs_rfecv_optimum` or `fs_shap_top*` could have silently resumed leaky pre-contract trials.

Code change: Optuna study names now include `FEATURE_CONTRACT`, e.g. `forecast_v1__fs_rfecv_optimum__LGBMRegressor__stage2_grid`. This keeps crash-resume semantics within the new contract while isolating the old `prog_leaky` search DBs.

Validation: `conda run -n energy-forecasting pytest tests/test_tuning.py -q` passed (14 tests), plus `py_compile` for `tuning.py`.

## 2026-07-13 — Stage A Feature Selection Launch

Before Stage A, `data/optuna` and `data/processed/datasets` were snapshotted to `data/retrain_checkpoints/forecast_fix_20260713/`.

First launch note: the initial detached command used `conda run` without `--no-capture-output`, which left the log empty and wrote the short-lived wrapper PID. That process group was terminated before relaunch.

Active Stage A launch:

```bash
setsid nohup bash -c 'echo $$ > logs/price_stage_a_feature_selection_20260713.log.pid; exec conda run --no-capture-output -n energy-forecasting energy-forecasting train price --feature-selection --use-rfecv --top-k 4' </dev/null > logs/price_stage_a_feature_selection_20260713.log 2>&1 &
```

- PID: 3050617
- Log: `logs/price_stage_a_feature_selection_20260713.log`
- PID file: `logs/price_stage_a_feature_selection_20260713.log.pid`
- Detach verification: `PPID=1`, `SID=3050617`, `TT=?`.
- Initial log shows reuse of the regenerated clean `price_max.parquet` and `price_slim.parquet` datasets.
