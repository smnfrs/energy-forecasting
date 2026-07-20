# Price Ensemble Construction Fix

> **⚠️ SUPERSEDED (2026-07-20) by [`ep_fidelity_reproduction_plan.md`](ep_fidelity_reproduction_plan.md).**
> The bakeoff / fair-OOS / floor-variant / **no-refit** design below was a
> divergence from EP and has been reverted. Production now follows EP verbatim:
> a category floor (2 per model family) plus inverse-MAE weights fit on the
> **recent holdout** — no method selection, no member zeroed, weights refit each
> retrain on fresh data. The "do NOT refit on the holdout" decision recorded
> here was coherent only for a fixed backtest and no longer holds. Retained for
> historical record only; do not implement from this document.

## Decision: do NOT refit production weights on the holdout (SUPERSEDED — see banner)

**Decided.** Production weights are fit on OOF predictions, evaluated on the
untouched holdout, and deployed exactly as fitted. We do **not** refit on
`OOF + holdout`. This keeps one clean contract:

- `metrics.mae` is the holdout MAE from weights fit on OOF.
- `conformal_quantile` is calibrated on residuals from that same fixed
  ensemble.
- `ensemble.weights` are exactly the weights used for both reported metrics and
  inference.

Rationale:

- **Conformal calibration is the clincher.** The PI quantile is calibrated on
  holdout residuals of a specific ensemble. Refitting the weights on
  `OOF + holdout` would force one of two bad outcomes: keep the old quantile and
  ship intervals calibrated on a model we don't deploy (coverage guarantee gone),
  or recalibrate on the refit ensemble — whose weights now saw the holdout — so
  the calibration residuals are in-sample and intervals come out too tight
  (systematic under-coverage in production). The clean OOF-fit / holdout-eval /
  holdout-calibrate chain avoids both.
- **The recency payoff is marginal here.** The selected method is `inverse_mae`
  (near-uniform ~1/8 weights); refitting on `OOF + holdout` nudges those weights
  by a rounding error. And in a re-blend the base models aren't retrained on the
  latest data anyway, so the part of staleness that actually matters isn't served
  by moving the weights.
- **Known trade-off:** production weights therefore exclude the most recent
  holdout window. In a fast-drifting regime that is a real (if small, given
  near-uniform weights) cost — it is the first thing to revisit if the daily
  forecast degrades.

If recency tracking is later shown to matter, capture it properly — a rolling
OOF window or a two-window select/calibrate split (see Deferred follow-ups) —
never by mixing holdout labels into production weights.

## Problem

The current price ensemble selection can overfit the holdout. In
`energy_forecasting/modeling/ensemble.py`, weight-based methods are fitted on
`preds_holdout, y_holdout` and then scored on that same holdout. The winner is
selected by lowest holdout MAE. This gives high-capacity methods such as
`slsqp_optimized` an in-sample advantage and can collapse the ensemble to a
thin blend with whole model families effectively zeroed.

The current `models/ensemble_config.json` shows the symptom: `slsqp_optimized`
is selected, with XGBoost and CatBoost weights effectively zero despite those
families being valuable in earlier Stage 5c runs.

This is a construction bug, not necessarily a base-model data bug. The fix is
to make method selection honest while keeping blends cheap and adaptive.

## User decisions

- Leave stacking as low priority for now. Stacking has never been selected, and
  full stacker persistence/export/inference support is a larger separate job.
- Keep single-model rows visible/selectable as a useful signal. If a single
  base model beats every blend, that is evidence of an ensemble candidate
  problem worth surfacing.
- Add minimum-weight variants of sparse methods, because blends are cheap and
  the goal is still to find the best holdout model under fair evaluation.
- Compute category-floor metrics directly from aligned OOF predictions,
  including both MAE and RMSE.
- Make OOF and holdout alignment validation a hard precondition inside the
  ensemble builder.
- Split retrain behavior into explicit modes instead of one ambiguous recompute
  path.
- Add tests for the new routing, floors, selection, alignment, config semantics,
  and retrain modes.

## Target design

Add one production orchestrator in `energy_forecasting/modeling/ensemble.py`:

```python
build_production_ensemble(model_runs)
```

The orchestrator should:

1. Load and validate OOF and holdout predictions for all candidate base models.
2. Compute per-model OOF MAE and RMSE from those aligned predictions.
3. Select a category-floored final candidate set.
4. Compare ensemble methods by fitting on OOF and evaluating on holdout.
5. Select the winning method from the fair comparison.
6. Calibrate conformal intervals from the selected ensemble's holdout residuals.
7. Return the selected ensemble, selection metrics, conformal metrics, comparison
   table, and final member set for config serialization and MLflow logging.

### Alignment preconditions

Before scoring or fitting, assert:

- OOF indexes are unique for every model.
- Holdout indexes are unique for every model.
- Common OOF and holdout indexes are sorted and non-empty.
- `y_true` values agree across all models on the common OOF index.
- `y_true` values agree across all models on the common holdout index.
- No duplicated rows are introduced by index intersection.
- Intersection loss is bounded: fail if the common OOF or holdout index drops
  more than the configured maximum fraction of rows, so accidental tiny
  intersections cannot produce plausible-looking metrics.
- The final OOF row count and holdout row count are logged and stored in
  comparison metadata.

Fail fast if any invariant is broken. Do not silently continue with suspicious
alignment.

### Category-floored candidate selection

Rename `BLEND_CATEGORY_MATCHERS` to `ENSEMBLE_CATEGORY_MATCHERS` and use it to
group candidates into:

- `linear`
- `lgbm`
- `xgboost`
- `catboost`

For each available category, keep the best OOF-MAE model and the best OOF-RMSE
model. If both metrics select the same model, keep one. Missing categories
should be logged explicitly; whether missing categories are fatal should be a
parameter with strict mode available for full training.

This selection happens after final validation/retrain artifacts exist, because
the floor is based on aligned OOF predictions. The current pre-retrain
per-model-type 20% `cv_mae` pruning may remain only as a cheap training-cost
prefilter; it is not the production ensemble candidate-selection rule. The
category floor guarantees representation in the final candidate set. It does
not by itself guarantee nonzero weights, so minimum-weight method variants are
added separately.

### Fair method comparison

Rewrite `compare_ensemble_methods` so all production methods fit on OOF and
score on holdout. Remove the current `HOLDOUT_FIT_METHODS` path for production
selection.

Keep the existing method registry, but separate methods into:

- **Production weight methods:** directly deployable from a weights dict.
- **Diagnostic-only stackers:** retained in the comparison table for signal,
  but not eligible for production selection until stacker persistence and
  inference support exist.
- **Single-model baselines:** retained and selectable as a warning signal.

Because `price_inference.py`, `model_store.py`, and `publish.py` currently
assume `ensemble.weights`, do not allow a stacker to become the production
config in this fix.

The comparison table must make this mechanical, with at least `method_kind`,
`fit_window`, `metric_window`, and `eligible_for_production` columns. Selection
filters to `eligible_for_production=True` before ranking by holdout MAE.

### Minimum-weight variants

Add floor variants for methods that can zero whole families:

- `slsqp_floor_2pct`
- `greedy_floor_2pct`
- `hill_climbing_floor_2pct`
- `simulated_annealing_floor_2pct`

Implementation rule: after fitting a sparse method, apply a minimum positive
weight to each selected candidate, then renormalize. A 2% floor is a starting
point; keep the value configurable as `ENSEMBLE_MIN_MEMBER_WEIGHT`.

Floor variants are scored only after the floor has been applied. Store both the
pre-floor and post-floor weights in metadata so the comparison score and the
deployed weights are auditable.

Do not apply a floor to models outside the category-floored candidate set.
`simple_average`, `inverse_mae`, and `inverse_rmse` already keep all candidates
positive and do not need separate floor variants.

## Config semantics

The new `models/ensemble_config.json` should make provenance explicit:

```json
{
  "ensemble": {
    "method": "inverse_mae",
    "weights_fit_window": "oof",
    "weights": {}
  },
  "metrics": {
    "mae": 16.9,
    "metric_window": "holdout",
    "selection_fit_window": "oof",
    "candidate_selection_window": "oof",
    "conformal_calibration_window": "holdout",
    "selection_metric": "mae"
  },
  "conformal_quantile": 29.0,
  "pi_coverage": 0.90,
  "models": []
}
```

Do not report any in-sample refit-on-holdout score as the production holdout
metric. If a later experiment refits on `OOF + holdout`, store that separately
from the honest selection metric.

## Retrain and reblend modes

Replace the current ambiguous `_recompute_ensemble` behavior with two explicit
paths:

- **Reweight current production members:** retrain or reload the current
  production member set, fit weights on OOF, evaluate on holdout, update weights
  if the degradation guard passes.
- **Full reselection / reblend:** rebuild the category-floored candidate set
  from all available candidate runs, run fair method selection, and emit a new
  production config.

The existing `energy_forecasting/deploy/retrain.py` call is currently broken
because it calls `select_best_ensemble(results, preds_holdout, y_holdout)` even
though `select_best_ensemble` accepts only the comparison dataframe. Fixing this
should be part of the retrain-mode cleanup.

## Code changes

### `energy_forecasting/modeling/ensemble.py`

- Add `select_final_models(...)` using OOF-derived MAE/RMSE per category.
- Add `validate_prediction_alignment(...)` for OOF/holdout invariants.
- Add `build_production_ensemble(...)` as the shared orchestrator.
- Rewrite `compare_ensemble_methods(...)` to fit production methods on OOF and
  score on holdout.
- Keep stackers in diagnostics but exclude them from production selection until
  deploy support exists.
- Add minimum-weight floor variants.
- Update docstrings that currently describe holdout fitting as EP parity.

### `energy_forecasting/modeling/price.py`

- Replace the per-model-type 20% pruning plus bakeoff block with
  `build_production_ensemble(model_runs)`.
- Rename MLflow tags from `ensemble_bakeoff` / `bakeoff` to
  `ensemble_selection`.
- Log the full comparison table, selected candidate set, alignment row counts,
  and config provenance fields.

### `energy_forecasting/config/modeling.py`

- Rename `BLEND_CATEGORY_MATCHERS` to `ENSEMBLE_CATEGORY_MATCHERS`.
- Rename `BLEND_DEGRADATION_THRESHOLD` to `ENSEMBLE_DEGRADATION_THRESHOLD`.
- Delete dead `BLEND_CANDIDATES_PER_CATEGORY` and
  `BLEND_CANDIDATES_RANDOM_POOL` comments until candidate reselection actually
  uses them.
- Add `ENSEMBLE_FINAL_PER_CATEGORY = 2`.
- Add `ENSEMBLE_MIN_MEMBER_WEIGHT = 0.02`.
- Document `ENSEMBLE_METHODS` as a fair OOF-fit / holdout-eval candidate set.

### `energy_forecasting/deploy/retrain.py`

- Replace the broken `_recompute_ensemble` signature path.
- Add explicit reweight vs reselection modes.
- Update degradation-threshold imports and comments away from SLSQP-specific
  language.

### Downstream consumers

- Keep deployed configs weight-based in this fix so
  `deploy/price_inference.py`, `deploy/model_store.py`, `deploy/publish.py`,
  API metadata, dashboard metadata, SHAP attribution, and narrative generation
  remain structurally compatible.
- Update user-facing method defaults from `slsqp_optimized` to the selected
  method or a neutral fallback such as `inverse_mae`.

## Repo-wide documentation and naming sweep

The point of this pass is that a future contributor (or model) should never find
a pointer that contradicts the shipped construction. After the code changes land,
every reference to the old bakeoff / `BLEND_*` / `slsqp_optimized`-as-default
vocabulary must be reconciled or explicitly marked historical.

### In-code

- `ensemble.py`, `price.py`: module- and function-level docstrings describe the
  fair-OOS + category-floor construction. No remaining comment may claim
  in-sample holdout fitting is "EP parity" — correct the docstrings on
  `fit_inverse_mae` and the module header, stating the fair-OOS rationale and
  citing this doc.
- `config/modeling.py`: comments on `ENSEMBLE_METHODS` describe it as the
  fair OOF-fit / holdout-eval candidate set (drop any "evaluated on holdout, best
  selected" wording that implies in-sample).

### `docs/`

- **This file** (`docs/bug-fixes/ensemble_construction_fix.md`) is the definitive
  write-up. Cross-link it from `docs/bug-fixes/README.md` (create/append) and from
  the coverage-remediation plan/log so the two threads connect.
- `docs/stage5_model_training.md` — update the canonical ensemble methodology
  section to describe the fair-OOS + category-floor construction as production.
- `docs/stage5c_*.md` (results / review / status, ×3) — **do not rewrite** the
  dated analysis; add a one-line banner at the top of each: *"Superseded (2026-07):
  the SLSQP bakeoff described here was replaced — see
  `docs/bug-fixes/ensemble_construction_fix.md`."*
- `docs/master_plan.md` — Stage 5c summary line and the **Bug fixes** section:
  replace the "SLSQP 5-model 11.148" framing with the corrected construction and
  the honest number (~16.9 on the Apr–Jul holdout).
- `docs/mlflow_conventions.md` — update the `ensemble_step` tag value
  (`bakeoff` → `ensemble_selection`).
- `docs/roadmap.md` — adjust ensemble-related backlog items.
- `docs/source_repo_guide.md` — enrich the EP `blend.py` pointer
  (`select_final_models` + inverse-MAE) as the reference construction.

### Project-level

- `CLAUDE.md` — the status line's "5 models: LGBM 34.6%…" SLSQP description is
  obsolete; restate as the category-floored, fair-OOS-selected construction and
  the current forecast-fix status.
- Memory: update `[[project_ensemble_diversity_collapse]]` to record the adopted
  fix (fair OOS selection + category floor, no holdout refit), so recall doesn't
  re-surface the diagnosis as if unresolved.

### Naming unification (single direction)

Standardize vocabulary on **"ensemble"**; retire stray `BLEND_*`:

- `BLEND_CATEGORY_MATCHERS` → `ENSEMBLE_CATEGORY_MATCHERS`
- `BLEND_DEGRADATION_THRESHOLD` → `ENSEMBLE_DEGRADATION_THRESHOLD`
- delete dead `BLEND_CANDIDATES_PER_CATEGORY`, `BLEND_CANDIDATES_RANDOM_POOL`
  and their misleading `select_candidates()` comment
- add `ENSEMBLE_FINAL_PER_CATEGORY`, `ENSEMBLE_MIN_MEMBER_WEIGHT`
- MLflow tags `ensemble_step="bakeoff"` / `dataset_name="ensemble_bakeoff"`
  → `"ensemble_selection"`
- `api/routes.py` default `"slsqp_optimized"` → `"inverse_mae"`

Closing grep (archive excluded) must return only intentional historical mentions:

```
grep -rn "BLEND_\|slsqp_optimized\|bakeoff\|select_candidates" \
  energy_forecasting docs CLAUDE.md
```

## Tests

Update `tests/test_ensemble.py` and related deploy/API tests to cover:

- All production methods fit on OOF and evaluate on holdout.
- Stackers are diagnostic-only for production config selection.
- Single-model rows remain visible and selectable.
- Category-floored selection computes OOF MAE and OOF RMSE directly from
  aligned predictions.
- Missing categories are logged or rejected according to strict-mode setting.
- Duplicate OOF or holdout indexes fail before fitting.
- `y_true` mismatches across models fail before fitting.
- Minimum-weight methods enforce the configured floor and renormalize.
- Config metrics distinguish fit window, metric window, and conformal
  calibration window.
- Retrain reweight mode and full reselection mode call the intended paths.

## Verification

1. Run focused tests:
   `conda run -n energy-forecasting pytest tests/test_ensemble.py tests/test_deploy_model_store.py tests/test_deploy_price_inference.py tests/test_api.py -q`
2. Run lint:
   `conda run -n energy-forecasting ruff check`
3. Reblend from existing price model runs without retraining base models.
4. Inspect `models/ensemble_config.json`:
   - method is weight-based, not stacker-only;
   - `weights_fit_window` is `oof`;
   - metric window is `holdout`;
   - at least one candidate per available family was considered;
   - floor-method weights respect `ENSEMBLE_MIN_MEMBER_WEIGHT` if selected.
5. Run a local forecast dry path:
   `energy-forecasting deploy forecast --skip-update`
6. Search for stale terminology:
   `grep -R "BLEND_\\|ensemble_bakeoff\\|bakeoff\\|SLSQP" energy_forecasting docs CLAUDE.md`
   and leave only intentional historical/archive references.

## Deferred follow-ups

- Full stacker deployment support: persist the chosen meta-learner, export/load
  it with release artifacts, and run inference through it.
- Recency-weight experiments: rolling OOF windows, two-window selection and
  calibration, or carefully measured `OOF + holdout` refit variants.
- Feature-set diversity beyond family diversity: replace nested SHAP prefixes
  with structurally distinct candidate sets.
- Multi-window honest evaluation so a single difficult holdout does not become
  the only model-quality story.
