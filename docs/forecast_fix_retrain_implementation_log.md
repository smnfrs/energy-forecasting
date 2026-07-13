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
