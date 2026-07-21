# Forecast Fix Coverage Remediation Implementation Log

**Started:** 2026-07-14
**Plan:** `forecast_fix_coverage_remediation_plan.md`

## 2026-07-14 — Guarding and Pre-flight Implementation

Implemented the non-training parts of Phase G/P5 first so the backfill and follow-on
price retrain cannot silently reuse fallback-sourced holdouts:

- Added shared forecast source labeling/count helpers in `features.forecast_inputs`.
- Added `features.forecast_coverage` with exact price-holdout coverage assertion and
  monthly historical-forecast artifact coverage checks.
- Wired the price training pipeline to assert coverage over the exact final dataset
  holdout split before tuning/blending/conformal calibration.
- Wired daily deploy to run the monthly historical-forecast coverage monitor after
  gen/load historical forecasts are updated and before price inference starts. The
  existing `--no-price` path still skips this check, so the remediation window can
  continue publishing gen/load-only outputs.
- Added `scripts/forecast_fix_coverage_tools.py` with `spans` and `coverage`
  diagnostics for repeatable pre-/post-backfill checks.
- Updated `scripts/forecast_fix_dataset_audit.py` to use the shared source-label
  helper and the exact `price_max` holdout split when available.

Validation:

```bash
conda run --no-capture-output -n energy-forecasting ruff check   energy_forecasting/features/forecast_inputs.py   energy_forecasting/features/forecast_coverage.py   energy_forecasting/modeling/price.py   energy_forecasting/deploy/inference.py   scripts/forecast_fix_dataset_audit.py   scripts/forecast_fix_coverage_tools.py   tests/test_forecast_inputs.py   tests/test_forecast_coverage.py   tests/test_deploy_inference.py
# All checks passed

conda run --no-capture-output -n energy-forecasting pytest   tests/test_forecast_inputs.py   tests/test_forecast_coverage.py   tests/test_deploy_inference.py   tests/test_price.py -q
# 17 passed
```

## 2026-07-14 — Fold Count Decision: 233, not 235

The plan's initial `235` fold recommendation was corrected during pre-flight. The
first span diagnostic version used stale generated gen/load datasets and reported an
old 2026-03-27 end. Raw inputs were then checked directly:

- `data/processed/tso/*.parquet` reaches 2026-06-30 11:00 UTC.
- `data/raw/weather/*/*/history.parquet` reaches 2026-06-29 23:00 UTC.
- `data/raw/weather/*/*/hist_forecast.parquet` reaches 2026-06-30 23:00 UTC.

After fixing the span diagnostic to model training rows from actual weather and
CV/holdout rows from `hist_forecast`, `235` folds would have started OOF at
2021-12-21, before hist-forecast weather starts on 2022-01-01. That violates the
backtest-honest test-fold assumption.

Decision: set `GEN_LOAD_HISTORICAL_FOLDS = 233` and keep
`GEN_LOAD_MAX_TRAIN_HOURS = 51_000`. Pre-flight then reported, for every target and
region:

- OOF/export start: 2022-01-04 00:00 UTC
- OOF end: 2026-06-22 23:00 UTC
- holdout/export end: 2026-06-29 23:00 UTC
- `export_start_has_hist_forecast_coverage = true`

This preserves the 2022-01-15+ price-training target while avoiding pre-coverage
hist-forecast rows.

## 2026-07-14 — Current Coverage Baseline and Snapshot

`python scripts/forecast_fix_coverage_tools.py coverage` reproduced the known hole:

- 2026-03 own fraction: 645/744 = 86.69%
- 2026-04 own fraction: 20/720 = 2.78%
- 2026-05 own fraction: 0/744 = 0.00%
- 2026-06 own fraction: 10/720 = 1.39%
- `price_max` exact holdout source mix: own=54, smard=2106, actual=0, missing=0
  over 2026-04-03 00:00 → 2026-07-01 23:00.

Pre-flight `--reuse-params` source-run check found all 48 required base runs:
`found=48 missing=0`.

Snapshot before overwrite:
`data/retrain_checkpoints/forecast_fix_coverage_20260714/historical_forecasts_pre_backfill/`.

## 2026-07-14 — Gen/Load Backfill Launch

Launched the Phase G backfill detached:

```bash
setsid nohup bash -c 'echo $$ > logs/gen_load_backfill_20260714.log.pid; exec conda run --no-capture-output -n energy-forecasting energy-forecasting train gen-load --reuse-params --parallel 4' </dev/null > logs/gen_load_backfill_20260714.log 2>&1 &
```

Launch verification:

- PID file: `logs/gen_load_backfill_20260714.log.pid`
- PID: `3219921`
- Detach state: `PPID=1`, `SID=3219921`, `TT=?`
- Log: `logs/gen_load_backfill_20260714.log`
- Initial log shows Wave 1/3 started, best configs loaded from existing MLflow runs,
  51,000-row datasets/test-fold datasets being written, and new generation MLflow
  runs starting.

Monitor commands:

```bash
ps -o pid,ppid,sid,tty,etime,%cpu,%mem,cmd -p 3219921
tail -n 120 logs/gen_load_backfill_20260714.log
```

Post-run gates to execute before Phase P:

```bash
conda run --no-capture-output -n energy-forecasting python scripts/forecast_fix_coverage_tools.py coverage
conda run --no-capture-output -n energy-forecasting python scripts/forecast_fix_dataset_audit.py
```

Then compare rebuilt `data/processed/historical_forecasts/` against the snapshot at
`data/retrain_checkpoints/forecast_fix_coverage_20260714/historical_forecasts_pre_backfill/`
and preserve/re-append any newer live rows if the backfill export end is behind the
snapshot/live artifact end.

## 2026-07-15 — Backfill Completion and Post-run Gates

The detached gen/load backfill completed successfully. Final log line:

- `Done: 64 succeeded, 0 failed`

The rebuild overwrote `data/processed/historical_forecasts/*` with OOF+holdout rows
ending 2026-06-29 23:00 UTC. As planned, live rows from the pre-backfill snapshot
were merged back:

- Source snapshot: `data/retrain_checkpoints/forecast_fix_coverage_20260714/historical_forecasts_pre_backfill/`
- Merged: 168 newer rows per artifact, all 20 artifacts
- Final artifact span: 2022-01-04 00:00 UTC -> 2026-07-07 11:00 UTC
- Final rows per artifact: 39,480

Post-merge coverage diagnostics:

- 2026-03: own=744/744 = 100.00%
- 2026-04: own=720/720 = 100.00%
- 2026-05: own=744/744 = 100.00%
- 2026-06: own=708/720 = 98.33%, smard=12, actual=0, missing=0
- 2026-07: own=24/24 = 100.00%
- Exact `price_max` holdout source mix: own=2148, smard=12, actual=0, missing=0
  over 2026-04-03 00:00 -> 2026-07-01 23:00, own fraction 99.44%.

`python scripts/forecast_fix_dataset_audit.py` passed:

- All eight `forecast_*` columns have 0 NaN.
- Own residual identity max absolute error: 0.0.
- Dataset schema audit found no `prog_`, `pct_prog_`, or `prognostiziert` tokens.

Decision: Phase G is complete. The remaining 12 SMARD fallback rows in the exact
price holdout are below the release preference threshold issue level (`own >=99%`,
actual=0, missing=0) and are accepted for proceeding to Phase P.

## 2026-07-15 — Stage P Launch

Reconfirmed the post-backfill coverage gate immediately before price retraining:

- Exact `price_max` holdout source mix: own=2148, smard=12, actual=0, missing=0
  over 2026-04-03 00:00 -> 2026-07-01 23:00.
- Own fraction: 99.44%, clearing the configured guard (`>=95%`) and the preferred
  release threshold (`>=99%`), with no actual or missing fallback rows.

Cleared stale Stage A artifacts so the corrected historical forecasts must be used:

- Removed `data/processed/datasets/price_*.parquet`.
- Removed stale `data/optuna/fs_*.db` studies from the corrupted feature-selection
  and tuning pass, preventing silent reuse of pre-remediation Optuna state.

Launched the Stage P price retrain detached:

```bash
setsid nohup bash -c 'echo $$ > logs/price_stage_p_retrain_20260715.log.pid; exec conda run --no-capture-output -n energy-forecasting energy-forecasting train price --feature-selection --use-rfecv --top-k 4' </dev/null > logs/price_stage_p_retrain_20260715.log 2>&1 &
```

Launch verification:

- PID file: `logs/price_stage_p_retrain_20260715.log.pid`
- PID: `3537654`
- Detach state: `PPID=1`, `SID=3537654`, `TT=?`
- Log: `logs/price_stage_p_retrain_20260715.log`
- Initial log shows the feature-selection pipeline started and `price_max` dataset
  regeneration loaded `data/processed/merged.parquet`.

## 2026-07-15 — Stage P Feature Selection and Guard Check

The full feature-selection stage completed after a long RFECV phase. The RFECV
worker pool was verified active while the log was quiet. Results:

- RFECV curve saved: `data/processed/datasets/price_fs_rfecv_curve.parquet`
- RFECV optimum: 65 features, `cv_mae=25.210`, `holdout_mae=18.036`
- Selected top-4 candidates for tuning:
  - `fs_shap_top96`: `cv_mae=25.889`, `holdout_mae=17.430`
  - `fs_shap_top99`: `cv_mae=25.592`, `holdout_mae=17.552`
  - `fs_shap_top130`: `cv_mae=26.531`, `holdout_mae=17.651`
  - `fs_shap_top90`: `cv_mae=26.151`, `holdout_mae=17.721`

Materialized corrected subset datasets:

- `data/processed/datasets/price_fs_shap_top96.parquet`
- `data/processed/datasets/price_fs_shap_top99.parquet`
- `data/processed/datasets/price_fs_shap_top130.parquet`
- `data/processed/datasets/price_fs_shap_top90.parquet`

The exact holdout forecast-source guard passed for all selected datasets:

- own=2148, smard=12, actual=0, missing=0
- own fraction: 99.44%

Tuning then started on `fs_shap_top130`, with fresh `forecast_v1__...` Optuna
studies.


## 2026-07-15 — Stage P Completion and Model Decision

Stage P price retraining completed successfully and wrote a new production
`models/ensemble_config.json`. Final ensemble method: `slsqp_optimized`.

Holdout metrics over 2026-04-03 00:00 -> 2026-07-01 23:00 (2,160 rows):

- MAE: 16.2841 EUR/MWh
- RMSE: 35.4615 EUR/MWh
- R2: 0.7723
- Conformal quantile: 29.3755
- PI coverage: 90.05%
- PI width: 58.7511 EUR/MWh

Non-zero ensemble members:

- `LGBMRegressor__fs_shap_top130`: 0.7391196519
- `LGBMRegressor__fs_shap_top99`: 0.1117870809
- `Ridge__fs_shap_top130`: 0.1118757586
- `Ridge__fs_shap_top99`: 0.0372175086

Decision: accept the honest Stage P ensemble for release. The old 11.148 MAE was
from the contaminated holdout and is retained only as historical context, not as a
valid comparison target.

## 2026-07-15 — Validation and Stress Slices

Added `scripts/forecast_fix_price_validation.py` to reconstruct the production
ensemble holdout predictions from MLflow prediction artifacts, compare against a
4-week seasonal baseline, and write a JSON report. Latest report:
`docs/forecast_fix/forecast_fix_price_validation_20260715.json`.

Validation summary:

- All holdout: model MAE 16.2841 vs baseline MAE 34.2894, skill 52.51%
- High-solar top decile: model MAE 28.1225 vs baseline MAE 55.2933, skill 49.14%
- Negative-price hours: model MAE 26.7704 vs baseline MAE 68.0904, skill 60.68%
- Residual-load ramp top decile: model MAE 16.3068 vs baseline MAE 36.9268, skill 55.84%

Final dataset/coverage gates:

- Exact price holdout source mix: own=2148, smard=12, actual=0, missing=0
- Own fraction: 99.44%
- All eight `forecast_*` columns: 0 NaN
- Price dataset schema audit: no `prog_`, `pct_prog_`, or `prognostiziert` tokens

## 2026-07-15 — Export, Packaging, and Workflow Re-enable

Export fixes implemented in `energy_forecasting.deploy.model_store`:

- `export_price_models()` now prunes stale `models/price/*.joblib` artifacts.
- `export_price_models()` now regenerates `models/price_feature_cols.json` from the
  exact production feature-version parquet schemas.
- `models/price_feature_cols.json` now contains only the active production feature
  versions: `fs_shap_top130` and `fs_shap_top99`.
- The stale `fs_rfecv_optimum`/`fs_shap_top90`/`fs_shap_top247` feature manifest,
  including banned pre-remediation `prog_` tokens, was replaced.

Exported production artifacts:

- Price: 4 joblibs, matching the four non-zero ensemble members.
- Gen/load: 16 joblibs, refreshed from the current `models/gen_load_config.json`
  after an initial local smoke exposed stale exported gen/load artifacts.
- Release archive rebuilt as `models.tar.gz` (234M) with `models/price/`,
  `models/gen_load/`, `models/ensemble_config.json`, `models/gen_load_config.json`,
  `models/price_feature_cols.json`, and the two production price datasets.

Workflow changes:

- `.github/workflows/daily_forecast.yml` re-enables price inference by removing
  `--no-price`.
- `.github/workflows/daily_forecast.yml` re-enables AI narrative generation.
- `.github/workflows/retrain.yml` packages the two active price datasets
  (`price_fs_shap_top130.parquet`, `price_fs_shap_top99.parquet`) and removes the
  stale dataset paths.

## 2026-07-15 — Smoke and Regression Checks

Controlled price inference smoke on local delivery date 2026-07-01 passed:

- 24 rows, 2026-07-01 00:00 -> 2026-07-01 23:00
- `fs_shap_top130`: 24/24 complete feature rows
- `fs_shap_top99`: 24/24 complete feature rows
- Mean prediction: 167.5357 EUR/MWh
- SHAP attribution present: `has_shap=True`

During the first smoke, Ridge SHAP attribution failed because exported Ridge folds
are sklearn Pipelines, not bare estimators. Fixed `deploy/shap_attribution.py` to
apply pipeline preprocessing before reading the final linear model coefficients,
and updated `tests/test_deploy_shap_attribution.py` to cover pipeline-backed Ridge
folds.

A live default `deploy forecast --skip-update` smoke was intentionally not accepted
as the final release gate on this local checkout because `data/processed/merged.parquet`
ends at 2026-07-01 while the current D+1 delivery date is 2026-07-16. The strict
live D+1 forecast-artifact check correctly failed on missing local future artifacts.
The daily CI path should first run the data update step, then gen/load inference,
then the same strict price check.

Regression checks passed:

```bash
conda run --no-capture-output -n energy-forecasting pytest tests/test_forecast_inputs.py tests/test_forecast_coverage.py tests/test_deploy_inference.py tests/test_price.py tests/test_deploy_model_store.py tests/test_deploy_shap_attribution.py -q
# 21 passed

conda run --no-capture-output -n energy-forecasting ruff check energy_forecasting/config/modeling.py energy_forecasting/deploy/inference.py energy_forecasting/deploy/model_store.py energy_forecasting/deploy/shap_attribution.py energy_forecasting/features/forecast_inputs.py energy_forecasting/features/forecast_coverage.py energy_forecasting/modeling/price.py scripts/forecast_fix_dataset_audit.py scripts/forecast_fix_coverage_tools.py scripts/forecast_fix_price_validation.py tests/test_forecast_inputs.py tests/test_forecast_coverage.py tests/test_deploy_inference.py tests/test_deploy_model_store.py tests/test_deploy_shap_attribution.py
# All checks passed
```
