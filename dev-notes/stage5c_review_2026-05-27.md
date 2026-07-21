# Stage 5c read-only review — 2026-05-27

Reviewed while another Claude instance was actively running `train price --quick`
(PID 533376, mid-Ridge linear grid ~trial 115/360). No exceptions in the log;
process is healthy. Findings ordered by severity.

## Significant deviations from the plan

### 1. Feature-selection candidates never feed back into training
`feature_selection.run_feature_selection` produces SHAP-curated subsets
(`shap_top40`, `rfecv_optimum`, …) and logs them to `price/feature_selection`,
but `price.run_price_pipeline` only iterates over the hard-coded
`{slim, full, max}` set (`price.py:58-62`, `price.py:236-239`). The whole
point of §5c.2 — that linear models prefer larger sets while trees prefer
SHAP-curated — is bypassed. The CLI also exposes no `select-features` command
(plan §5c "Files Summary" lists this on `cli.py`).

### 2. Ensemble PI calibration is missing
Plan §5c.4 Step 4 requires "Calibrate ensemble prediction intervals
(post-hoc conformal)", and the documented `ensemble_config.json` schema
includes `conformal_quantile`. `run_price_pipeline` writes only point-metric
stats; `ensemble_config_dict` (`ensemble.py:436-470`) has no PI fields.
`intervals.post_hoc_conformal` exists (from 5a) but is never called for the
ensemble.

### 3. Step 4's candidate-selection layer is collapsed
The plan describes `select_candidates` (6 per category: 2 incumbents,
1 best-MAE, 1 best-RMSE, 2 random) → `validate_candidates` (24×5 fits) →
`select_final_models` → `train_and_blend`. `price.py:260-277` skips all of
that and retrains every `(model_type × feature_version)` winner into the
ensemble pool. With `--quick` that's only 3 models, but with the full matrix
it's 18, not 24, and there's no random/RMSE/incumbent reasoning at all.
Resulting `ensemble_config.json` also drops most of the per-model fields
documented in the plan (`selection_reason`, `holdout_mae`, `dataset_path`,
`feature_list`, `preprocessing` block).

### 4. Required §5d artifacts not logged from feature selection
The plan's "MLflow Artifact Logging Requirements" demand SHAP values
per-feature/sample and the full RFECV `n_features → score` curve as run
artifacts. `_log_candidate` (`feature_selection.py:272-297`) logs only
`feature_list.json` + scalar metrics. The data exists in memory
(`shap_ranking`, `_curve` from `rfecv_select`) — it just never reaches MLflow.
The 5d notebooks can't be built without it.

## Correctness / robustness concerns

### 5. Ridge linear grid is mostly numerical garbage
Live log shows Ridge CV-MAE bouncing between ~20 and **1e16 → 1e22**. Best
~20 is found early (trial 74), then ~80% of subsequent trials are pathological.
Two causes from `search_spaces.py:176-196`:
- `target_transform="log_shift"` is in the grid for Ridge/Lasso, but the
  kill-switch in `tuning.py:367` only short-circuits ElasticNet. Power
  prices go strongly negative; `log_shift` after an unsigned shift produces
  extreme inverse-transform errors.
- `yeo_johnson` rows also explode (preproc_idx 16-23 in log).

This won't break selection but wastes ~10 min and creates hundreds of useless
MLflow runs per linear model. Worth gating like the ElasticNet check, or
pre-filtering `LINEAR_PREPROCESSING_GRID` for Ridge/Lasso.

### 6. `compute_neg_price_stats` uses `min_periods=1`
(`market.py:182-184`). During the first 30/90 days the rolling fraction is
computed from a partial window — biased. The `d1` lag pushes the first 24h to
NaN (which `dropna()` removes), but rows 25-720 still carry partial-window
estimates that survive the warm-up drop. Most rolling stats in this codebase
NaN until the window fills; this one is the odd one out. Either intentional
("never let this be NaN") or an oversight — confirm with the author.

### 7. `prepare_price_dataset` re-writes the parquet after `prepare_dataset` already logged provenance
(`price.py:103-111`). The MLflow dataset hash recorded by `prepare_dataset`
no longer matches the file on disk after the `dropna` overwrite. Won't fail
anything but the provenance link is silently wrong.

### 8. Stale CatBoost defaults
`_make_model` (`tuning.py:71-74`) doesn't set `verbose=0` or
`allow_writing_files=False`. Once the full run kicks in (not the `--quick`
skip), expect log spam and `catboost_info/` directories per worker process.

## Minor / cosmetic

- `comparison.drop(columns=[]).to_dict(...)` (`price.py:317`) — `.drop` is a no-op.
- `_fetch_predictions` (`price.py:161-165`) uses the deprecated
  `client.download_artifacts`; `mlflow.artifacts.download_artifacts` is the
  replacement.
- `_stack_predictions` doesn't guard against an empty `common_oof`
  intersection — would silently produce empty preds/y if a model_run has no
  overlapping index.

## What's correct and worth keeping

- 5c.0 wiring (engine branches for `forecast_*`, `eeg_regime`, `neg_price_*`,
  per-tech `pct_prog_*`; SHORT_NAMES; availability rules) all lines up, and
  `tests/test_5c_derivations.py` covers the cases that matter including
  tz-naive merged-data alignment.
- `gen_load_forecasts.load_gen_load_forecasts` is clean — pre-flight
  `FileNotFoundError`, residual derivation pulls inputs automatically, tz
  normalisation handled both ways.
- `TrackedRun` tag validation passes for every call site checked;
  `EXPERIMENTS["price_model_training"]` / `"price_production"` both exist.
- Optuna SQLite resume is wired with `load_if_exists=True`; the DBs at
  `data/optuna/slim__*` will let an interrupted run pick up where it left off.

The smoke run looks like it will complete successfully — the structural issues
(#1–#4) are about whether the *full* pipeline does what the plan says, not
whether the current invocation crashes.
