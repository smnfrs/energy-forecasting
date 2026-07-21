# Forecast Fix — Coverage-Hole Remediation & Re-Evaluation Plan

**Date:** 2026-07-14
**Depends on:** `forecast_fix.md` (leakage fix, landed), `forecast_fix_retrain_plan.md`
(cleanup/retrain, partially executed)
**Status:** Diagnosis complete. Remediation not started. **The price retrain
completed on 2026-07-14 must NOT ship** — its holdout was corrupted by the coverage
hole described below.

---

## 0. Why this plan exists

The leakage fix (`forecast_fix.md`) and the first price retrain
(`forecast_fix_retrain_plan.md`) both completed. The retrained ensemble reported a
holdout **MAE of 15.77 vs the old leakage-inflated 11.148** — a ~4-point jump far
larger than the "few tenths" the same change cost upstream. Investigation showed the
degradation is **almost entirely an evaluation artifact**, not a model regression.

### Diagnosis (evidence)

1. **The own gen/load forecasts are good — better than SMARD's.** On the dense
   2022→2026-03 overlap, residual-load forecast error vs actuals:
   - own (our models): **MAE 2,641 MW**
   - SMARD `prognostizierte`: MAE 5,094 MW
   Load forecast is ~2% MAE and stable over time. The forecasts are not the problem.

2. **There is a 3-month coverage hole in the own historical-forecasts artifacts.**
   All-5-artifact coverage by month:
   ```
   2022-01 … 2026-02 : 100%
   2026-03           :  87%
   2026-04           :   3%   ← hole
   2026-05           :   0%   ← hole
   2026-06           :   2%   ← hole
   2026-07           : 100%
   ```
   The dense 218-fold OOF batch (Phase A, 2026-05-07) stopped at ~2026-03-27; the
   daily appender only resumes in July. April–June was never generated. **This is
   produced by the gen/load pipeline and was not touched by the leakage fix.**

3. **The price holdout fell straight into the hole.** The last-90-day holdout is
   **2026-04-02 → 2026-07-01**, which `build_forecast_columns` fills 97% from SMARD
   fallback (56 own rows of 2,161). Own vs SMARD residual differ by **5,124 MW MAE**
   (corr 0.89) — a large feature shift on the single most important price driver.

4. **Confirmed by re-scoring the retrained ensemble** (OOF-based, method validated
   by reproducing the 15.772 control on the holdout):

   | Window | Feature source | Ensemble MAE |
   |---|---|---|
   | Holdout 2026-04→07 (reported) | 97% SMARD | **15.77** |
   | Last 90 own-days, ending 2026-03-27 | own | **12.51** |
   | 2025 full year | own | 13.55 |

   On data that represents production (own forecasts), the ensemble is ~12.5, not
   15.77. The model trains on own and deploys on own, but was **evaluated, blended,
   and conformal-calibrated on SMARD** — so the ensemble weights and conformal
   quantile are fit against the wrong distribution.

### Consequence

- Nothing is wrong with the leakage-fix code or the gen/load models.
- The 2026-04→06 coverage hole must be filled so the price holdout is own-covered
  and represents production.
- The current retrained ensemble's weights/conformal/metric are contaminated and
  must be redone against a corrected holdout (feature selection likely too — see
  §P2).

---

## 1. Current state (what is already done)

From `forecast_fix.md` and `forecast_fix_retrain_plan.md`:

- ✅ Leakage fix implemented (`forecast_*` contract, strict D+1 inference, guards).
- ✅ Phase 0 code cleanup + `feature_contract=forecast_v1` tag plumbing.
- ✅ Preservation exports + MLflow archive-in-place (`archived=true`,
  `feature_contract=prog_leaky`) for all price runs. Nothing deleted.
- ✅ Price datasets regenerated clean (0 banned tokens). Parity audit passed.
- ✅ Optuna study isolation (`forecast_v1__…`).
- ⚠️ Price retrain ran to completion (~918 model runs + bakeoff + reblend) **but
  against the corrupted holdout** — do not ship; `models/ensemble_config.json` is
  uncommitted and the implementation log stops at Stage A.
- ✅ Daily price step disabled (`deploy forecast --no-price`) — keep disabled until
  release.
- ❌ §7 export never ran: `models/price_feature_cols.json` (46 `prog_` tokens) and
  `models/price/*.joblib` are still the old leaky production artifacts.

---

## 2. Phase G — gen/load backfill (close the hole)

Rebuild the dense OOF historical-forecasts so coverage spans 2022-01-15 → 2026-07-01
with no gap. Uses the purpose-built cheap path — **no Optuna re-search**.

### G1. Config change
`energy_forecasting/config/modeling.py`:
- `GEN_LOAD_HISTORICAL_FOLDS` **218 → 233** (OOF span backwards from the 2026-07-01
  data end; 233 weeks preserves the full 2022-01-15 start while keeping the earliest
  CV test rows after `hist_forecast` weather starts on 2022-01-01. The initial
  235-fold draft was rejected by pre-flight because current inputs ending
  2026-06-29 would push first OOF rows into 2021-12. 218 would start around
  late March 2022 and drop early-2022 rows.)
- `GEN_LOAD_MAX_TRAIN_HOURS` **48_000 → 51_000** (optional; holds the per-fold
  sliding train window near ~474 days instead of shrinking to ~355. Data exists back
  to 2015, so there is headroom.)

### G2. Pre-flight
- `--reuse-params` errors if any `(target, region, model_type)` combo lacks a prior
  finished run. Confirm all generation experiments are intact first (they are marked
  VALID in `forecast_fix_retrain_plan.md §1`, but verify).
- Snapshot `data/processed/historical_forecasts/` before overwriting.
- Dry-calculate the expected OOF + holdout export span for each target/region before
  the long run. The nominal `233` folds should preserve the 2022-01-15 start, but
  the true span depends on each dataset's post-cleaning end timestamp, the 7-day
  gen/load holdout carve, and any dropped rows. Print the expected first/last
  exported timestamp per combo and confirm the national five-artifact intersection
  will cover 2022-01-15 → 2026-07-01.

### G3. Run (detached, per `~/.claude/CLAUDE.md`)
```bash
setsid nohup bash -c 'echo $$ > logs/gen_load_backfill_20260714.log.pid; \
  exec conda run --no-capture-output -n energy-forecasting \
  energy-forecasting train gen-load --reuse-params --parallel 4' \
  </dev/null > logs/gen_load_backfill_20260714.log 2>&1 &
disown
```
Scope: all per-TSO base models (reuse-params) + stacking ensembles + national
aggregation, in the enforced wind/solar → load → gen_load_diff order. Expect a few
hours, not the 8–12h full search. Verify detachment (`PPID=1`, own `SID`).

### G4. Verify coverage
- All-5-artifact own coverage ≈ 100% across 2026-04 → 2026-07 (re-run the coverage
  diagnostic).
- Continuity: no structural break at the 2026-03-27 seam (the old OOF end) — plot
  `forecast_load` / `forecast_residual_load` across it.
- `build_forecast_columns(merged)` holdout source mix now **own-dominated** (was
  own=56 / smard=2,106).
- Preserve or re-append any live rows that existed beyond the new OOF+holdout export
  max timestamp. `_export_historical_forecasts()` overwrites each parquet from the
  selected run's OOF + holdout artifacts, so compare against the G2 snapshot and
  merge back rows newer than the rebuilt artifact end if needed. Alternatively, run
  fresh gen/load inference before re-enabling price.
- Smoke-test the next live D+1 strict path:
  `build_forecast_columns(extended_df, strict_index=d1_index)` must pass with all
  five own artifacts present. This catches workflow-order or post-backfill live-row
  gaps before price inference is re-enabled.

**Gate:** do not proceed to Phase P until the last-90-day holdout window is
own-sourced at the selected threshold (default: ≥95%, with `actual=0` and
`missing=0`). Prefer ≥99% for release unless any remaining SMARD fallback rows have
a documented benign cause.

---

## 3. Phase P — price re-evaluation (redo what the hole corrupted)

### P1. Regenerate price datasets
- `rm -f data/processed/datasets/price_*.parquet` (mandatory — the cache
  short-circuits before validation; see `forecast_fix_retrain_plan.md §4`).
- Regenerate; confirm 0 banned tokens and that the holdout rows now carry own
  forecasts.

### P2. Re-run feature selection
The Stage A selection (`fs_shap_top75/78`, `fs_corr_filtered`, `fs_shap_top250`) was
scored partly against the corrupted holdout. Re-run selection on the corrected
datasets so importances/selections reflect own-forecast features. (Per-model Optuna
hyperparameters were CV-based on the dense region and are probably still valid, but
re-selecting is cheap insurance.)

Current CLI behavior note: `energy-forecasting train price --feature-selection
--use-rfecv --top-k 4` does **not** stop after feature selection; it continues into
tuning, final retraining, ensemble bakeoff, and conformal calibration. Treat that
single command as covering P2+P3 unless a dedicated stop-after-selection /
pin-feature-version checkpoint path is added first.

### P3. Re-train / re-blend / re-calibrate
- Re-run the price bakeoff (or at minimum re-fit the SLSQP ensemble weights and the
  conformal quantile) against the corrected own-covered holdout.
- Ensemble weights are fit on holdout (project pattern) → **must** be redone.
- Conformal quantile is fit on holdout residuals → **must** be redone.

### P4. Verify the honest number
- Holdout MAE on the corrected (own-covered) holdout. Do **not** use the leaky
  11.148 as a release target. Use it only as historical context. The relevant
  checks are: own-covered holdout, improvement over a naive seasonal baseline,
  sane stress-regime behavior, and consistency with the own-covered references
  already measured (`12.51` last-own-90 and `13.55` for 2025).
- Sanity-check vs a naive seasonal baseline and on stress regimes (high-solar /
  negative-price / ramps) per `forecast_fix_retrain_plan.md §6`.
- Frame honestly: the leaky 11.148 is **not** the target to beat; the corrected
  own-covered number is the new baseline.

### P5. Add guards (prevent silent recurrence)
- **Holdout coverage assertion:** price training/eval refuses to blend or report a
  holdout whose exact training split is not sufficiently own-covered. Compute the
  source labels over the same holdout indices produced by `carve_holdout()`, not an
  approximate `index.max() - 90 days` slice. Default gate: own ≥95%, actual=0,
  missing=0; prefer own ≥99% for release unless a documented exception is accepted.
- **Artifact coverage monitor:** a check that flags gaps in
  `data/processed/historical_forecasts/*` (e.g. any month below N% all-5 coverage),
  wired into the daily deploy after gen/load inference updates historical forecasts
  and before price inference starts, so a future hole is caught immediately.

---

## 4. Phase R — release (from `forecast_fix_retrain_plan.md §7`)

Only after P4 passes:
1. Commit the corrected ensemble + update the implementation log.
2. Run the §7 export: refit/export winners → `models/price/`, rewrite
   `price_feature_cols.json`; verify **zero** `prog_`/`pct_prog_` tokens and that its
   feature versions match `ensemble_config.json`.
3. Rebuild `models.tar.gz` / GitHub release.
4. Re-enable the price step in `daily_forecast.yml`; verify one clean run publishes
   all expected JSON.

## 5. Phase C — cleanup (from `forecast_fix_retrain_plan.md §2b, §8`)

Only after Phase R has one clean daily run:
- Prune superseded `price/model_training` artifacts (`mlflow gc`), keeping metadata
  and `price/production` fallbacks.
- Update `docs/mlflow_conventions.md` and `CLAUDE.md` status.

---

## 6. Acceptance gates

- [ ] Own coverage ≈ 100% across 2026-04 → 2026-07; no seam break at 2026-03-27.
- [ ] Price holdout window meets the explicit own-source gate (default ≥95%,
      `actual=0`, `missing=0`; prefer ≥99% for release).
- [ ] Regenerated price datasets + exported `price_feature_cols.json` contain zero
      banned tokens; feature versions match `ensemble_config.json`.
- [ ] Corrected holdout MAE beats the naive seasonal baseline and is consistent
      with own-covered reference windows; stress regimes sane. The leaky 11.148 is
      context only, not a release gate.
- [ ] Holdout-coverage assertion + artifact-coverage monitor in place.
- [ ] First post-release daily run publishes all price JSON.

---

## 7. Sequenced checklist

- [ ] G1 config: `GEN_LOAD_HISTORICAL_FOLDS` 218→233 (+ `MAX_TRAIN_HOURS` 48k→51k)
- [ ] G2 pre-flight: verify prior gen/load runs exist; snapshot historical_forecasts;
      dry-calculate expected export spans
- [ ] G3 run `train gen-load --reuse-params --parallel 4` detached + PID saved
- [ ] G4 verify coverage + seam continuity + holdout own-source mix; preserve or
      re-append newer live rows; strict D+1 smoke test
- [ ] P1 delete stale `price_*.parquet` + regenerate + token check
- [ ] P2/P3 re-run feature selection command, noting it continues through tuning,
      retrain, reblend, and conformal calibration
- [ ] P4 verify honest MAE + stress regimes + baseline framing
- [ ] P5 add holdout-coverage assertion + artifact-coverage monitor
- [ ] R commit + §7 export + verify tokens/schema + models.tar.gz + release
- [ ] R re-enable price step; verify clean daily run
- [ ] C prune MLflow bulk; update conventions + CLAUDE status

---

## 8. Open decisions

1. **Folds 233 vs 235 vs 218** — 233 is selected after pre-flight: it preserves
   the 2022-01-15+ target while keeping first OOF rows after `hist_forecast`
   weather starts on 2022-01-01. 235 reaches before hist-forecast coverage with
   current inputs; 218 drops early-2022 rows. *Decision: 233.*
2. **Re-run feature selection (P2) or reuse the current selection** — re-running is
   cheap insurance against holdout contamination in the selection step. *Default:
   re-run.*
3. **Coverage-guard strictness** — default to own ≥95%, `actual=0`, `missing=0`
   over the exact training holdout split. Prefer own ≥99% for release unless any
   remaining SMARD fallback rows have a documented benign cause. *Needs confirming.*
