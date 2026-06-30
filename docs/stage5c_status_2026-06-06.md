# Stage 5c status — 2026-06-06

Verified the live code against the 2026-05-27 review and the 2026-05-29
results doc, then root-caused the LGBM underperformance flagged in the
results doc. This file is the current source of truth for 5c state.

## Code state vs the 2026-05-27 review

Re-read `price.py`, `tuning.py`, `search_spaces.py`, `training.py`. The major
structural items from the review are resolved as of the 2026-05-29 run:

| Review item | Status |
|---|---|
| #1 Feature-selection candidates never feed training | **Fixed** — `run_price_pipeline(use_feature_selection=True)` ranks candidates and tunes the top-k subsets (`price.py:380-421`). |
| #2 Ensemble PI calibration missing | **Fixed** — post-hoc conformal in `price.py:501-523` (q=25.18, coverage 90.05%). |
| #5 Ridge/Lasso `log_shift`/`yeo_johnson` blow-ups | **Fixed** — `TARGET_TRANSFORMS = ["none"]` (`search_spaces.py:182`); transforms no longer in any grid. |
| #3 Candidate-selection layer collapsed | **Fixed (2026-06-30)** — per-model-type pruning before retrain: within each model type, configs whose cv_mae exceeds the type's best by >20% are dropped. Cross-family diversity (linear vs tree) is untouched. (`price.py`, after tuning loop). |
| #4 §5d artifacts (SHAP/RFECV curves) not logged | **Fixed (2026-06-30)** — SHAP ranking + cutoff curve saved to `data/processed/datasets/price_fs_shap_*.parquet` and logged to a `meta_shap` MLflow run; RFECV curve saved to `price_fs_rfecv_curve.parquet` and attached as artifact to the `rfecv_optimum` run. (`feature_selection.py`). |
| #6 `compute_neg_price_stats` `min_periods=1` | **Fixed (2026-06-30)** — changed to expanding initially (`min_periods=1`) then rolling windows; early rows are numeric and depth is average positive below-zero magnitude. (`market.py:182-184`). |
| #8 CatBoost `allow_writing_files` unset | **Fixed (2026-06-30)** — `allow_writing_files=False` added to `CatBoostRegressor(...)` in `tuning.py`. No more `catboost_info/` spam. |

`max` is still a live linear-only candidate (`price.py:421`) and target
transforms, though pinned to `none`, still carry dead `_LogShiftTransformer`
/ `PowerTransformer` code in `training.py`.

## LGBM underperformance — root cause (investigated 2026-06-06)

The 2026-05-29 run had LGBM zero-to-trivially weighted in the ensemble
(0.004) and 0.3–1.3 MAE behind XGB on every feature set. That is wrong for a
model that should be ~on par with XGB. Two **config asymmetries**, both
confirmed empirically on `price_fs_rfecv_optimum` (n=40, 73,464-row pool,
3-fold expanding CV, no sample weights; scripts in `scripts/investigate_lgbm*.py`):

### 1. `num_leaves` pinned at 31 across the whole grid
`PRICE_TREE_WEIGHT_PROBE["LGBMRegressor"]` sets `num_leaves=31`, and
`PRICE_TREE_GRID["LGBMRegressor"]` overrides only
`learning_rate/max_depth/reg_alpha/n_estimators` — never `num_leaves`. So
`num_leaves` stays 31 even at `max_depth=10`. 31 leaves ≈ a fully-grown
depth-5 tree, while XGB (`max_depth=8`, depth-wise) and CatBoost (`depth=8`,
symmetric) grow to ~256 leaves. LGBM was running at ~8× less capacity.

### 2. `objective="mae"` (L1) degrades LightGBM
L1 has a zero hessian; LightGBM's histogram split-finding and Newton-style
leaf updates degrade under it (XGB's `reg:absoluteerror` is purpose-built and
handles it well). Switching LGBM to L2 (`objective="regression"`) gives the
best cv_mae of any tree config.

### Evidence (holdout = last 90 days)

| config | cv_mae | hold_mae |
|---|---|---|
| LGBM probe (mae, num_leaves=31) — **current** | 21.74 | 12.31 |
| LGBM mae, num_leaves=255, max_depth=-1 | 21.83 | 11.85 |
| **LGBM L2, num_leaves=255, max_depth=-1** | **20.22** | **11.76** |
| XGB probe (max_depth=8) — reference | 21.94 | 11.49 |
| CatBoost probe (depth=8) — reference | 21.81 | 13.55¹ |

¹ CatBoost here has no sample-weight pinning; it did better in the real run.
`objective="huber"` blew up (40+ MAE) on the default delta — not pursued.

**Conclusion.** LGBM was crippled by capacity starvation + an L1 objective,
not by anything intrinsic. Fixing both makes it competitive with XGB and
restores it as a genuine ensemble contributor — so the results-doc suggestion
to *drop* LGBM was treating a symptom. Same likely applies to keeping Lasso:
the fix is to make each family well-specified, then let the ensemble weight it.

## Planned changes (this session)

1. **Fix LGBM** — switch objective to L2, scale `num_leaves` with `max_depth`
   (and unbind depth), enable `bagging_freq` so `subsample` is not a no-op.
2. **Drop `max`** as a feature candidate everywhere.
3. **Remove target transforms** — delete the dead `log_shift`/`yeo_johnson`
   code paths and the grid axis (already pinned to `none`).
4. **Keep Lasso and LGBM.**
5. **Diversity/accuracy experiments** — add a robust non-tree model
   (HuberRegressor) for ensemble diversity; re-run the full pipeline and
   compare the ensemble against the 11.24 baseline.

## 2026-06-07 — changes applied + experiment launched

### Code changes
- **LGBM fix** (`search_spaces.py`): probe `num_leaves` 31→255 + `subsample_freq=1`;
  `PRICE_TREE_GRID["LGBMRegressor"]` now sets `num_leaves = 2^max_depth − 1`
  per config. Objective stays MAE (EP comparability) — L2-across-all-trees is
  now a `docs/roadmap.md` item (Price Models §).
- **Two diversity candidates added** (user: "test both and see"):
  `LGBMQuantile` (objective=quantile, α=0.5) registered as a 4th tree family;
  `Huber` (HuberRegressor, max_iter=3000) as a 3rd linear family
  (`LINEAR_ALPHA_GRID["Huber"]=[0.1,1.0,10.0]` — tiny α diverges).
- **Dropped `max`** feature set everywhere (`price.py`).
- **Target transforms** dropped as a search axis: `LINEAR_PREPROCESSING_GRID`
  is now a 2-tuple `(scaler, weight_half_life)`; `target_transform` pinned to
  `"none"` in `tuning.py`. The `_LogShiftTransformer`/`PowerTransformer` code
  in `training.py` is **kept** — gen/load models still use it.
- **`precomputed_datasets` param** added to `run_price_pipeline` so an
  experiment can reuse existing fs_* parquets and isolate model changes.

### Pre-run probes (on `price_fs_rfecv_optimum`, n=40, 90d holdout)
| candidate | hold_mae | err_corr vs XGB | read |
|---|---|---|---|
| XGB reference | 11.49 | — | — |
| **LGBM fixed (mae, leaves=255, bag1)** | **11.89** | 0.983 | fix worked — was 0.3–1.3 behind, now ~0.4; **but** near-duplicate of XGB |
| LGBMQuantile (α=0.5) | 12.03 | 0.983 | redundant with LGBM/XGB; little diversity |
| Huber (any α 0.1–10) | 17.3 | 0.832 | most diverse, but too weak; CV diverges on early folds at *every* α (not an α problem) |

Prediction: the LGBM capacity fix restores it as a genuine contributor;
LGBMQuantile and Huber are unlikely to earn meaningful ensemble weight. The
full SLSQP bake-off is the definitive test.

### Stale-Optuna-study contamination (caught + fixed)
First launch reused the 2026-05-29 Optuna studies (same `feature_version`
names, persisted in `data/optuna/*.db`). `GridSampler` saw the grids as
"exhausted" and returned cached trials computed with the **old**
`num_leaves=31` LGBM configs — so weight/grid *selection* was stale even
though `_train_winner` retrains fresh. Killed it, archived all 34 affected
studies to `data/optuna/archive_pre_5c_diversity_20260607/`, relaunched fully
fresh.

### Experiment running
`scripts/run_5c_diversity_experiment.py` over the 4 fs_* datasets, writing
`models/ensemble_config_5c_diversity.json` (baseline `ensemble_config.json`
@ MAE 11.24 untouched). Launched detached 15:21 →
`logs/5c_diversity_rerun_20260607_152137.log` (pid file alongside). Models:
LGBM, LGBMQuantile, XGB, CatBoost, Ridge, Lasso, Huber.

## 2026-06-24 — diversity experiment results

Re-ran with fresh Optuna studies (first launch 2026-06-07 crashed on empty
`feature_version` tag; fixed in `price.py:562`; second launch 2026-06-24
completed clean).

**Ensemble: MAE 11.148, RMSE 18.094, R²=0.844, PI coverage 90.05%, width 48.89**
(baseline 11.239 / 18.000 / 0.846 / 90.05% / 50.35)

Effective contributors (non-zero SLSQP weight):

| model | weight |
|---|---|
| LGBMRegressor__fs_shap_top90 | **0.346** ← LGBM fix confirmed |
| XGBRegressor__fs_rfecv_optimum | 0.197 |
| XGBRegressor__fs_shap_top90 | 0.188 |
| CatBoostRegressor__fs_rfecv_optimum | 0.184 |
| Ridge__fs_shap_top247 | 0.084 |

- **LGBMQuantile: zero weight** on all 4 feature sets — pre-run probe prediction confirmed.
- **Huber: zero weight** on all 4 feature sets — confirmed too weak / too divergent on early folds.

LGBM configs: `num_leaves` now 63–511 depending on `max_depth` (vs 31 in baseline).

## 2026-06-30 — review items closed + pipeline cleanup

### Diversity experiment promoted to production
`models/ensemble_config.json` ← `models/ensemble_config_5c_diversity.json` (MAE 11.148).

### LGBMQuantile and Huber removed from pipeline
- `search_spaces.py`: `LGBMQuantile` removed from `PRICE_TREE_WEIGHT_PROBE` and `PRICE_TREE_GRID`; `Huber` removed from `LINEAR_ALPHA_GRID`.
- `tuning.py`: `_make_model` simplified (no `LGBMQuantile`/`Huber` branches); `PRICE_TREE_TYPES = (LGBMRegressor, XGB, CatBoost)`, `PRICE_LINEAR_TYPES = (Ridge, Lasso)`.
- Next run ~20% faster per dataset (11 fewer grid trials).

### All review items fixed (see table above)
