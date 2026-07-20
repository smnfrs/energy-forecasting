# Reproduce EP's inference/retrain approach exactly

## Context

A founding principle of the merge (master_plan.md:1079, decision #6 — the "reproduce
first" stage-gate) is: **follow EP exactly and reproduce it before introducing
improvements from a real baseline.** The recent ensemble work drifted from this. In trying
to "improve" the blend we added a method-selection **bakeoff** (15+ weighting methods,
floor variants, diagnostic stackers, fair-OOS selection, a reserved holdout with a
"no-refit" rule) that **EP never had** — and it backfired: it overfit the holdout, then the
"honest" repair left production **stuck on April-vintage models**, because our retrain
reuses frozen feature datasets (`find_dataset()` → a Jul-1 parquet) and fits weights on
stale OOF while EP rebuilds features on fresh data and weights on the recent tail.

Meanwhile EP — running the *simple* construction — posts **live-forward blend MAE 16–18**
through 2026 (Feb–Mar 16.3, Jul 11–19 18.0). Our reblend lands at 17.2–17.4 on the same
market, i.e. **we already match EP on accuracy**; what we lack is EP's *freshness* and its
*simplicity*. The fix is not a cleverer ensemble — it is to **go back to EP's approach
verbatim** for construction, weighting, and retrain-time data freshness, and to document
that faithfully.

This plan supersedes the design in `docs/bug-fixes/ensemble_construction_fix.md` (the
bakeoff / fair-OOS / floor-variant / no-refit design). That document is retained as
historical record but marked superseded.

---

## EP's approach (the target — reproduce verbatim)

Anchored in `~/projects/energy_prices/src/modeling/blend.py` and `src/deploy/`.

1. **Fixed feature recipe, fresh data.** `_load_datasets` (blend.py:109) loads each model's
   saved feature **pipeline** (transformer) from MLflow and applies
   `pipeline.fit_transform(merged_df)` to the **current** `merged` parquet. Feature
   *selection is frozen*; the *data* is always current. (blend.py:124,155)
2. **Category-floored candidate set.** `select_final_models`: best-MAE + best-RMSE per
   category (`BLEND_CATEGORY_MATCHERS` = linear / lgbm / xgboost / catboost),
   `BLEND_FINAL_PER_CATEGORY = 2`. If one model wins both metrics, EP keeps the
   second-best RMSE model when available, so the intended production set is up to
   8 distinct members (2 per populated category). No bakeoff.
3. **Rolling recent-holdout fit.** For each model, split the *fresh* dataset at
   `len(X) − holdout_days*group_size` (`BLEND_HOLDOUT_DAYS = 90`), **refit the model from
   scratch on the train portion**, predict the last-90-days holdout, record `holdout_mae`.
   (blend.py:426-455)
4. **Inverse-MAE weights, always.** `_compute_inverse_mae_weights`: `1/(mae+1e-6)`
   normalized. No method comparison, no optimizer, no member ever zeroed. Weights fit on
   the **recent 90-day tail** — this is what keeps EP current.
5. **Point forecasts only.** EP has **no conformal / prediction intervals**.
6. **Evaluation is live-forward.** Each daily forecast is scored against next-day actuals
   (`deploy/data/model_errors.json`). The `blend_config.json` `blend_mae` (currently
   14.93, `blend_me` +2.84) is the *in-sample recent-holdout* number — optimistic by
   EP's own convention; the honest number is the live-forward log.
7. **Biweekly retrain** (`retrain.yml` cron `1,15`) on data refreshed **daily**
   (`daily_forecast.yml` "forecast includes data update" → `data-latest` release).

---

## Our current divergences (what to revert)

| Concern | EP | Us (now) | Action |
|---|---|---|---|
| Construction | inverse-MAE over 2-per-category floor | 15-method bakeoff + floor variants + diagnostic stackers + fair-OOS select | **EP-exact in production**; bakeoff code retained but moved to a standalone diagnostic script (§4) |
| Weight-fit window | recent 90-day holdout tail | OOF (2015→Apr), **no-refit** | **Revert to EP** (recent tail) |
| Retrain data | rebuild features from fresh `merged` inline | `find_dataset()` loads frozen Jul-1 parquet | **Rebuild inline (EP)** |
| Base models at retrain | refit from scratch on rolling train | reuse frozen Phase P runs | **Refit on rolling window** |
| Reported metric | in-sample recent-holdout (labeled) | fair-OOS holdout (~17) + conformal | Adopt EP metric; keep live-forward as truth |
| Prediction intervals | none | conformal PI (dashboard/API/story depend on it) | **Keep** as documented deviation (decided, §5) |

---

## Plan

### 1. Blend construction → EP-exact in production (bakeoff code retained, not called)

The bakeoff **code stays** — `compare_ensemble_methods`, `select_best_ensemble`, the full
`METHOD_FACTORIES` (inverse_mae, simple_average, slsqp, `*_floor_2pct`, stackers,
single-model baselines), and the `method_kind`/`eligible_for_production` columns — along
with its existing `tests/test_ensemble.py` coverage. It simply **moves out of the
production path** into a standalone diagnostic script (§4). Nothing is deleted; less churn,
tests stay green.

In `energy_forecasting/modeling/ensemble.py`:
- **Production entry point becomes EP-exact.** `build_production_ensemble` collapses to:
  align predictions → `select_final_models` (best-MAE + best-RMSE per category, with
  EP's second-best-RMSE fallback when the same model wins both metrics) → **inverse-MAE
  weights fit on the recent holdout** (§2) → return the `WeightEnsemble`. It **must not**
  call `compare_ensemble_methods`/`select_best_ensemble` or emit a comparison table,
  eligibility filter, or single-model rows.
- Keep `compare_ensemble_methods` et al. as an **importable research module** for the
  script; add a module docstring stating they are diagnostic-only and not part of
  production.
- The numpy-serialization bug **disappears from production** for free: with no comparison
  table / `eligible_for_production` flag written into `ensemble_config.json`, only native
  metric scalars remain. Still coerce the few metric floats to native types defensively.
  (The script owns its own output serialization — §4.)

### 2. Weights fit on the recent 90-day tail (fixes "stuck in April" on the weighting side)

- Replace the OOF-fit / no-refit rule with EP's: fit inverse-MAE weights on each model's
  **last `HOLDOUT_DAYS` days** of the *fresh* dataset (for hourly price models:
  90 days = 2,160 hourly rows). Drop `weights_fit_window="oof"`; the window is the
  recent holdout, matching EP.
- This reverses the `ensemble_construction_fix` "no-refit" decision **on purpose** — that
  decision was coherent only for a fixed backtest. EP fits on the recent tail and evaluates
  live-forward; we adopt the same contract.

### 3. Retrain rebuilds features from fresh data (fixes "stuck in April" on the model side)

In `energy_forecasting/deploy/retrain.py`:
- Replace `_retrain_one_price_model`'s `find_dataset(f"price_{fv}")` (frozen parquet) with
  EP's `_load_datasets` pattern: **rebuild the feature matrix from the current `merged`
  parquet** using the fixed feature recipe. Create a shared historical-feature helper
  rather than calling the D+1-oriented inference helper directly: load current
  `merged.parquet`, build strict `forecast_*` columns, engineer the full
  `PRICE_FEATURES_MAX` matrix, select the persisted `price_feature_cols.json` columns
  for each feature version, attach `target_price__target`, drop invalid rows, and use
  that fresh matrix for training/holdout prediction. The column list is the frozen
  recipe; the row data must come from today's merged file, not from
  `data/processed/datasets/price_*.parquet`.
- Refit each base model **from scratch on the rolling train split** (all − last 90 days),
  predict the recent holdout, feed §1/§2. The training window then rolls forward with the
  data — no more April cut.
- **The deployed model is the all−90d model itself, deliberately** — as EP does (it saves
  the `X_train`-fit pipeline; there is no all-data refit). This is *load-bearing, not
  staleness to fix*: the recent-holdout weight fit is only valid because the models never
  trained on that holdout. Deploying an all-data model would make the weights in-sample and
  collapse the EP contract. The real recency lever is retrain **cadence** (how often the
  window rolls), not training-on-all-data — flag any "train on all data for deploy" idea as
  a contract-breaking change, not a free win.
- **Keep `collect_oof=True` in retrain even though production ignores OOF.** EP-exact weights
  fit on the holdout only, so production needs no OOF — but the comparison script (§4) fits
  methods on OOF and evaluates on holdout. Retain OOF emission for diagnostics; do not let
  someone "optimise it away" on seeing production ignore it.
- **Persist the full candidate universe as reusable configs, not as implicitly fresh
  artifacts.** Reselection (and richer method comparison) needs all configured candidate
  hyperparams/feature versions, but steady-state retrain only refreshes the positive-weight
  production members. Keep every candidate config in `ensemble_config.json` (flagged
  selected / with weight, as today) or a sibling `candidates` manifest. Run IDs are only
  considered fresh for candidates retrained in the current run; the full universe becomes a
  fresh comparable artifact set only after a bootstrap/reselection run that retrains every
  candidate first.
- Keep the **feature recipe frozen** (do NOT re-run SHAP/RFECV selection each retrain — EP
  doesn't; the transformer is fixed). This keeps retrain cheap enough to schedule.
- Split the correction into two explicit phases:
  - **One-time bootstrap/reselection:** rebuild the EP-faithful production member set
    from all configured candidate runs (category floor with the second-best fallback,
    then inverse-MAE).
  - **Steady-state retrain:** match EP's biweekly path by retraining the configured
    positive-weight production member set from scratch on fresh data and recomputing
    inverse-MAE weights. Do not silently reselect every retrain; only reselect when
    `needs_reselection` is raised or the user explicitly requests it. Diagnostic comparison
    after this path is fresh for production members only, not for sidelined candidates.
  Retain the degradation guard (`ENSEMBLE_DEGRADATION_THRESHOLD`, EP's
  `needs_reselection`).

### 4. Standalone method-comparison script (the bakeoff's new home)

New `scripts/ensemble_method_comparison.py` — a diagnostic/research tool, **not** wired
into any workflow:
- **Input (option 1 — decided): artifact sets, with freshness made explicit.** By default,
  read the current positive-weight production members from `models/ensemble_config.json`
  and load their OOF/holdout prediction parquets — exactly the `reblend.py` pattern already
  prototyped this session. After a steady-state retrain, only those production-member
  artifacts are guaranteed fresh. Full-universe comparison is valid as a fresh comparison
  only after a bootstrap/reselection run has retrained every candidate first; otherwise the
  script must label the set as mixed/stale and treat the output as historical diagnostics.
  The script does **no** training of its own and introduces **no** second training path.
- **Compute:** `stack_model_predictions` → `compare_ensemble_methods` (unchanged) over the
  full method registry, returning the ranked table (method, `method_kind`, `mae`, `rmse`,
  `eligible_for_production`).
- **Output:** pretty-printed table to stdout; optional `--csv` / `--markdown` dump and an
  optional MLflow log (experiment `price/ensemble_research`) so method performance can be
  tracked across retrains. The script owns this serialization (handles numpy coercion
  itself), keeping it out of the production config.
- **Optional flags:** `--run-ids`/`--config` to point at an arbitrary artifact set,
  `--scope production|all-candidates` to choose positive-weight members vs the full
  candidate manifest, and `--methods` to subset the registry. `--scope all-candidates`
  should warn unless the config records that the current artifact generation came from a
  bootstrap/reselection run. (A future `--rebuild-from-fresh` is explicitly out of scope.)
- **CLI:** expose as `energy-forecasting research ensemble-compare` (or leave as a
  `python scripts/...` entry) — thin wrapper, no production coupling.

### 5. Conformal PI — keep as the first documented deviation (decided)

EP has no PI; we keep ours. It is post-hoc relative to EP's point forecast: it changes
neither the weights nor the point forecast, and the dashboard, API, and story site all
consume it. Recorded as the single sanctioned departure from EP in the deviations register
(§ Docs). Because the same recent holdout fits the inverse-MAE weights **and** calibrates the
residuals, the intervals are doubly in-sample and will likely run **materially too narrow**
(live under-coverage), not merely "optimistic". Acceptable for v1, but §6's monitoring must
**track live-forward PI coverage** as the trigger to graduate a cleaner calibration (a
separate calibration window) — that is the natural first improvement to the PI layer. No
other code change beyond calibrating on the new recent-holdout residuals.

### 6. Evaluation & reporting → EP convention

**Principle: numbers live in regenerated artifacts, not in prose.** The stale `11.148` in
`CLAUDE.md` is exactly the failure mode to avoid — a point-in-time metric frozen into docs
and then treated as a standing truth. Any specific MAE cited in a document is a **dated
observation of a shifting regime**, never an acceptance criterion or target. Judgements are
made **relative to a live benchmark**, not an absolute constant, and we *want* these numbers
to fall as models improve — so no doc should canonise a current value as "good".

- The config's quoted MAE is the **in-sample recent-holdout** value (EP's `blend_mae`),
  written by the pipeline, **labeled optimistic**, and **timestamped** in the config — it is
  a regenerated artifact field, not a doc constant. The honest metric is the **live-forward**
  monitoring log (our Stage 7/8 error tracking = EP's `model_errors.json`).
- Retire the fixed Apr–Jul backtest as the headline. **Retire the leaky `11.148` everywhere
  it appears — and do not replace it with a new magic number.** Docs describe the *method*
  and *how to obtain the current number* (run the monitor / read the config / fetch EP's live
  `model_errors`), not the number itself.
- The standing benchmark is **EP's current live-forward errors, fetched at evaluation time**
  (gh-pages `model_errors.json`) — a moving comparison, not a band written down here.

---

## Code changes (files)

- `energy_forecasting/modeling/ensemble.py` — make `build_production_ensemble` EP-exact
  (floor + inverse-MAE on recent holdout); **retain** `compare_ensemble_methods`,
  `select_best_ensemble`, `METHOD_FACTORIES`, floor variants, stackers as a diagnostic
  module (docstring: "not production"). Defensive native-type coercion of production metrics.
- `energy_forecasting/modeling/price.py` — `build_production_ensemble` call simplified;
  stop logging the comparison table into the production run. **Rename the MLflow tag
  `ensemble_selection` → `ensemble_production`** (no selection happens in production now;
  leave `ensemble_selection` for the diagnostic script if anything). Keep "ensemble" as our
  term; note EP's is "blend".
- `energy_forecasting/deploy/retrain.py` — rebuild-from-fresh-`merged`; refit-from-scratch
  rolling split; one-time EP bootstrap/reselection plus steady-state EP retrain of the
  positive-weight production members; keep the degradation guard.
- `scripts/ensemble_method_comparison.py` — **new** standalone diagnostic (§4), defaulting
  to positive-weight production members and warning on full-candidate comparisons unless the
  artifact set was produced by a bootstrap/reselection run.
- `energy_forecasting/config/modeling.py` — **keep** `ENSEMBLE_METHODS`, floor-variant
  names, `ENSEMBLE_MIN_MEMBER_WEIGHT` (now consumed by the script), plus
  `ENSEMBLE_CATEGORY_MATCHERS`, `ENSEMBLE_FINAL_PER_CATEGORY=2`, `HOLDOUT_DAYS=90`,
  `ENSEMBLE_DEGRADATION_THRESHOLD`. Re-comment `ENSEMBLE_METHODS` as "diagnostic script
  registry, not production".
- `energy_forecasting/cli.py` — optional `research ensemble-compare` command.
- `energy_forecasting/api/routes.py` / `deploy/price_inference.py` / `model_store.py` —
  method string is always `inverse_mae`; verify weight-based config still loads.
- `energy_forecasting/deploy/model_store.py` / release workflow — ensure
  `ensemble_config.json`, exported price model files, `price_feature_cols.json`, and any
  uploaded release artifact all reference the same fresh run IDs after retrain. This
  lockstep check prevents "fresh config, stale model" skew.
- `tests/` — **keep** existing bakeoff/floor/stacker tests (the code still exists). Repoint
  production/retrain tests to EP-exact behaviour (category floor, inverse-MAE on recent
  holdout, rebuild-from-fresh-merged retrain, bootstrap vs steady-state retrain semantics,
  candidate-config vs fresh-run-id semantics, config/model/feature-column artifact lockstep,
  config serialization round-trip), and add smoke tests for the comparison script over both
  production-member and all-candidate fixture artifacts.

## Documentation cleanup

- **Supersede**: banner atop `docs/bug-fixes/ensemble_construction_fix.md` +
  `..._implementation_log.md` → "Superseded by `ep_fidelity_reproduction_plan.md` — the
  bakeoff/fair-OOS/no-refit design was a divergence from EP and was reverted."
- **New deviations register** in this doc / `docs/bug-fixes/README.md`: the *only* sanctioned
  departures from EP (currently: conformal PI). Everything else must match EP.
- `docs/stage5_model_training.md` — rewrite the ensemble section to EP-exact (floor +
  inverse-MAE on recent holdout; no bakeoff).
- `docs/master_plan.md` — Stage 5c line + Bug-fixes section: describe the EP-faithful
  construction and the reproduce-first correction; **remove the `11.148` claim without
  substituting a new number** — point to the monitoring log / config for the current value.
- `CLAUDE.md` — status line: replace the obsolete "11.148 (5 models…SLSQP)" with a
  *number-free* description of the EP-faithful construction and where the live number lives
  (monitoring / config), so the status line can't go stale again.
- `docs/source_repo_guide.md` — point at `blend.py` `_load_datasets`/`retrain_blend` as the
  reference for retrain-time feature rebuild + recent-holdout weighting.
- Memory: update `[[project_ensemble_diversity_collapse]]` and `[[project_ensemble_fit_pattern]]`
  to record that EP-exact (floor + inverse-MAE on **recent holdout**, rebuild-on-fresh-data)
  is the adopted approach; the bakeoff/no-refit was reverted.

## Verification

1. `conda run -n energy-forecasting ruff check` + `pytest tests/test_ensemble.py
   tests/test_deploy_retrain.py tests/test_deploy_price_inference.py tests/test_api.py -q`.
2. Reblend/retrain on **fresh** data → confirm `ensemble_config.json`: method
   `inverse_mae`, up to 8 members with 2 per populated category where available, positive
   normalized weights (none zeroed), and the holdout end equals the latest non-null
   `target_price` timestamp in the fresh merged dataset (not Jul 1) — proving the freeze
   is gone.
3. **Leakage-reproduction gate (run before trusting the fresh path).** Run the new rebuild
   helper on the *old* window (data through the Phase-P cutoff) and confirm it reproduces the
   existing frozen Phase-P holdout predictions **within tolerance**. This is a *relative*
   check against the prior artifact, not a target number: a large unexplained MAE **drop**
   vs the frozen predictions signals reintroduced `forecast_*`/`prog_*` leakage; a large
   divergence signals a feature-construction mismatch. Only trust the fresh-data path once it
   reproduces the frozen artifact.
4. Sanity-check the fresh config's recent-holdout MAE **relative to EP's current live
   `model_errors` fetched at the time** (not a written-down band) and to our own live-forward
   log — flag only if it is implausibly better (leakage) or worse (regression). No absolute
   pass/fail threshold is baked in.
5. `deploy forecast --skip-update` loads the config and runs inference (incl. conformal PI)
   without error.
6. Confirm artifact lockstep after export/release packaging: every production model name in
   `ensemble_config.json` has a matching exported model artifact for the same run ID, any
   scaler artifact needed by that run, and feature columns for its feature version in
   `price_feature_cols.json`; the release/upload payload contains the same files.
7. Grep confirms the **production** path (`price.py`, `retrain.py`) no longer calls
   `compare_ensemble_methods` / `select_best_ensemble`; the only remaining callers are
   `scripts/ensemble_method_comparison.py` and its test.
8. Run `scripts/ensemble_method_comparison.py` against the fresh production-member artifacts
   → prints the ranked method table and the CSV/MLflow outputs land where expected. Run it
   once with `--scope all-candidates` against a fixture/config marked as steady-state and
   confirm it warns about mixed/stale candidate artifacts; run it against a
   bootstrap/reselection-marked fixture and confirm it treats the full universe as fresh.

---

## Next steps (improvements — only *after* EP reproduction is green)

1. **Recency knob / two-window calibration** — once EP-faithful, revisit whether a proper
   rolling two-window split beats EP's in-sample-holdout weighting (the honest-eval idea,
   re-introduced as a *deliberate* improvement, not a silent divergence).
2. **Under-forecast bias** — EP shows +2.84, we're higher; check once base models are
   refit on fresh data (staleness may be most of our gap).
3. **Conformal as a first-class improvement** — kept (§5); formalize/measure its coverage
   against EP's point-only baseline as our first sanctioned deviation.
4. **Use the comparison script's evidence** — if it repeatedly shows a stacker/method
   beating EP's inverse-MAE out-of-sample across retrains, that becomes the *justified*
   case to graduate a method into production (the "improvement from a real baseline" the
   principle intends).
5. **Automated freshness cadence** — schedule the fresh-data retrain (tower or slimmed CI)
   so production stops depending on a manual Phase-P batch.
6. **skops retrain deserialization bug** and **Release/code-skew sequencing** (carried over).
