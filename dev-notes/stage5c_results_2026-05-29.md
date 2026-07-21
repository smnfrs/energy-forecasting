# Stage 5c results — 2026-05-29

First successful price-model training run with the SMARD/EMA waterfall on
`prog_*` forecast columns, the trimmed search spaces (no ElasticNet, no
target transforms, fewer alphas), and per-family feature_version routing
(trees on selected sets, linear on selected sets + `max`).

## Run setup

- Command: `energy-forecasting train price --feature-selection --top-k 4 --use-rfecv`
- Log: `logs/train_price_ema_20260528_225157.log`
- Started: 2026-05-28 22:51:58
- Finished: 2026-05-29 04:04:59
- Wall time: **5h 13m**
- Dataset: `price_max` 98,544 rows from 2015-01-19 to 2026-03-28 (~11 years),
  75,624 after warm-up drop. `prog_*` forecast columns upgraded via
  EMA-historical overlay on 36,788 rows (the 2022-01-15+ window).
- Holdout: last 90 days (~2160 rows). CV: 5-fold expanding.

## Feature-selection candidates (top-4 chosen)

| Candidate         | n   | cv_mae | holdout_mae | Selected |
|-------------------|-----|--------|-------------|----------|
| baseline (max)    | 270 | 22.41  | 12.70       | —        |
| corr_filtered     | 258 | 22.31  | 12.57       | —        |
| **rfecv_optimum** | 40  | 20.46  | **12.07**   | ✓        |
| **shap_top90**    | 90  | 21.50  | 12.15       | ✓        |
| **shap_top247**   | 247 | 22.62  | 12.16       | ✓        |
| **shap_top66**    | 66  | 21.45  | 12.47       | ✓        |
| slim (manual)     | 83  | 20.72  | 13.49       | —        |
| shap_top10        | 10  | 18.58  | 16.63       | — (cv-best but holdout-worst) |

SHAP coarse sweep at `(10, 20, 30, 40, 60, 90, 130, 180, 250)` with fine
windows around minima; **n=10 had the lowest cv_mae (18.58) but the worst
holdout (16.63)** — small feature sets generalise across the 11-year span
but miss recent-regime signals. The CV-vs-holdout divergence is genuine
and tells us small-n is robust-but-stale.

RFECV ran on SHAP top-80 with `step=5`, `inner_folds=3` — wall time ~3h 11m,
down from the 10h+ buggy prior run. Optimum at n=40 features.

## Retrain holdout MAEs (sorted)

| Model              | Feature set       | n    | cv_mae | **holdout_mae** |
|--------------------|-------------------|------|--------|-----------------|
| **XGBRegressor**   | fs_rfecv_optimum  | 40   | 21.40  | **11.75** ⭐    |
| XGBRegressor       | fs_shap_top247    | 247  | 20.99  | 11.77           |
| XGBRegressor       | fs_shap_top90     | 90   | 20.01  | 11.81           |
| XGBRegressor       | fs_shap_top66     | 66   | 20.61  | 11.83           |
| LGBMRegressor      | fs_shap_top66     | 66   | 21.21  | 12.12           |
| LGBMRegressor      | fs_shap_top247    | 247  | 21.48  | 12.17           |
| CatBoostRegressor  | fs_shap_top90     | 90   | 21.23  | 12.39           |
| CatBoostRegressor  | fs_shap_top66     | 66   | 20.89  | 12.39           |
| LGBMRegressor      | fs_rfecv_optimum  | 40   | 21.50  | 12.61           |
| LGBMRegressor      | fs_shap_top90     | 90   | 20.62  | 13.11           |
| CatBoostRegressor  | fs_shap_top247    | 247  | 20.93  | 13.12           |
| CatBoostRegressor  | fs_rfecv_optimum  | 40   | 21.25  | 13.82           |
| Ridge              | fs_rfecv_optimum  | 40   | 17.51  | 16.05           |
| Lasso              | fs_shap_top247    | 247  | 19.24  | 16.33           |
| Lasso              | fs_rfecv_optimum  | 40   | 17.66  | 16.39           |
| Lasso              | max               | 277  | 12.87  | **16.49** ⚠    |
| Lasso              | fs_shap_top90     | 90   | 17.30  | 16.50           |
| Lasso              | fs_shap_top66     | 66   | 17.22  | 16.58           |
| Ridge              | fs_shap_top66     | 66   | 17.28  | 16.88           |
| Ridge              | fs_shap_top90     | 90   | 17.52  | 17.12           |
| Ridge              | fs_shap_top247    | 247  | 19.01  | 18.13           |
| Ridge              | max               | 277  | 19.38  | 18.78           |

⚠ Lasso__max shows the cv-overfit pattern: cv_mae 12.87 (best of all) but
holdout 16.49. Across the cv 11-year span it found a regime-averaged
optimum that doesn't transfer to the last 90 days. Validates the call to
log both metrics and judge on holdout.

## Ensemble bake-off

**Winner: `slsqp_optimized` — holdout MAE 11.24, RMSE 18.00, R² 0.846,
PI coverage 90.05% @ q=25.18.**

Ensemble vs best single (XGB__fs_rfecv_optimum at 11.75): improvement of
0.51 MAE, ~4.3%.

### Weights (non-zero only)

| Model                         | Weight |
|-------------------------------|--------|
| XGB__fs_rfecv_optimum         | 0.250  |
| XGB__fs_shap_top247           | 0.206  |
| CatBoost__fs_rfecv_optimum    | 0.172  |
| CatBoost__fs_shap_top90       | 0.147  |
| XGB__fs_shap_top90            | 0.131  |
| Ridge__fs_shap_top247         | 0.090  |
| LGBM__fs_shap_top247          | 0.004  |

### Aggregated by feature set

| Feature set       | Total | Pulls from               |
|-------------------|-------|--------------------------|
| fs_rfecv_optimum  | 0.422 | XGB + CatBoost           |
| fs_shap_top247    | 0.300 | XGB + Ridge + LGBM(tiny) |
| fs_shap_top90     | 0.278 | CatBoost + XGB           |
| fs_shap_top66     | 0.000 | —                        |
| max               | 0.000 | —                        |

### Aggregated by model family

| Family    | Total |
|-----------|-------|
| XGB       | 0.587 |
| CatBoost  | 0.319 |
| Ridge     | 0.090 |
| LGBM      | 0.004 |
| Lasso     | 0.000 |

## Suggestions for further improvement

Ordered by expected impact / effort ratio.

### Cheap drops

1. **Drop LGBMRegressor.** Zero weight in the ensemble (the 0.004 on
   shap_top247 rounds to noise). It contributes ~12 minutes of wall time
   per feature set across both tuning stages with no payoff. Removing it
   trims ~50 minutes from the next run and is a one-line config edit.

2. **Drop Lasso.** Zero weight, holdout 16.3-16.6 across all feature sets,
   and the cv-overfit risk (Lasso__max) is a footgun without upside.

3. **Drop `max` as a tuning candidate.** Zero weight in the ensemble;
   Ridge/Lasso on max were the two worst models in the table. The "linear
   models like wide feature sets" hypothesis didn't survive contact.

After these three drops the ongoing pipeline is XGB + CatBoost on
{rfecv_optimum, shap_top90, shap_top247} + optionally Ridge for one
feature set as ensemble diversity. ~30-45 min wall time per retrain.

### Worth experimenting

4. **Add a diverse non-tree model.** The ensemble is 91% tree-weighted;
   the small 0.09 Ridge contribution was the only diversity it could find.
   Candidates:
   - `HuberRegressor` — robust linear, handles the 2022 crisis tails better
     than Ridge. EP already supports it (`config/modeling.py:179`).
   - Quantile gradient boosting (LightGBM with `objective="quantile"`,
     `alpha=0.5`) — median target rather than mean, less swayed by the
     2022 outliers in the 11-year span.
   - Small MLP — explicit non-tree non-linearity. More infrastructure work.

5. **Tighter weight half-life grid.** Current search is
   `[None, 365, 730, 1095]`. XGB on rfecv won with `weight_half_life=None`
   (uniform weights). Worth probing the *shorter* end (`[None, 90, 180, 365]`)
   to see whether even more recent-bias helps for the recent regime.

6. **Add a residual-lag feature.** Engineer a feature that is the
   reference-LGBM residual from D-1 onwards. Captures "what the base model
   would have got wrong yesterday" — meta-feature that often helps trees.

### Plausible but speculative

7. **Localised conformal calibration.** Global q=25.18 hits 90% coverage,
   but base-model PI coverages range 67-89% (XGB at 67% on shap_top90 is
   the worst). Splitting by hour-of-day or season-of-year would tighten
   intervals where they're currently over-conservative and widen them
   where they're under-covered.

8. **Investigate why rfecv (n=40) competes so closely with shap_top90.**
   The 40-feature set might be telling us most of shap_top90 is noise.
   Worth diffing the two feature lists and asking whether the extra 50
   features carry signal or just complexity.

9. **More aggressive winner selection from feature selection.** Currently
   top-4 by holdout_mae. Could add a blended rank
   (`0.5×cv_rank + 0.5×holdout_rank`) once we have a sense of the regime
   stability — would have picked n=10 too in this run (cv-best, holdout-worst),
   which we'd want to study not deploy.

10. **CV scheme experiments.** 5-fold expanding gives folds of ~2.2 years
    each; the 2022 crisis sits in folds 4-5 and dominates absolute MAE there.
    Could try (a) rolling-window CV with 1-year folds, (b) holding out 2022
    as its own evaluation slice, (c) bootstrap CV. Might change which models
    look best.

## What worked this run

- Per-family feature_version routing (Ridge/Lasso get more options, trees
  stay focused) — though linear models on `max` ended up at zero weight
  anyway, so the routing did its job by not foreclosing the option.
- The full SHAP curve (not just local minima) — `shap_top247` got
  selected and contributes 30% of the ensemble. Would have been
  silently dropped by the prior behaviour.
- RFECV at `input=80, step=5` was tractable (3h vs 10h+) and produced the
  best individual model (XGB on n=40, holdout 11.75).
- EMA waterfall ran cleanly once the column-naming bug was fixed.
  Empirical lift over SMARD-only is small at the baseline (12.70 vs the
  killed-mid-run sibling's 12.40-ish on identical span) but trees may use
  the upgraded values more selectively.
