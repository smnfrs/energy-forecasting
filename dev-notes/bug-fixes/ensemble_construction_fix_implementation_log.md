# Ensemble Construction Fix Implementation Log

> **⚠️ SUPERSEDED (2026-07-20) by [`ep_fidelity_reproduction_plan.md`](ep_fidelity_reproduction_plan.md).**
> This logs the (now reverted) bakeoff / OOF-fit / no-refit implementation. The
> bakeoff functions still exist in `modeling/ensemble.py` but only as a
> diagnostic module behind `scripts/ensemble_method_comparison.py`; production
> construction is EP-exact. Historical record only.

## 2026-07-17

### Scope

- Implement the fair OOF-fit / holdout-eval price ensemble construction from
  `docs/bug-fixes/ensemble_construction_fix.md`.
- Fold in review changes before code work:
  - comparison rows carry production eligibility metadata;
  - stackers remain diagnostic-only;
  - sparse floor variants are scored after floor application;
  - alignment validation rejects excessive index intersection loss;
  - config provenance includes fit, selection, metric, calibration, and row-count
    metadata;
  - retrain semantics are split between fixed-member reweighting and full
    reselection.

### Audit Trail

- Started from current code where weight methods fit on holdout in
  `compare_ensemble_methods`, production training logs `ensemble_bakeoff`, and
  retrain calls `select_best_ensemble` with the wrong signature.
- Implementation edits are expected in:
  - `energy_forecasting/config/modeling.py`
  - `energy_forecasting/modeling/ensemble.py`
  - `energy_forecasting/modeling/price.py`
  - `energy_forecasting/deploy/retrain.py`
  - downstream defaults/tests/docs touched by stale `slsqp_optimized`,
    `BLEND_*`, or `bakeoff` semantics.

### Config Update

- Replaced `BLEND_*` constants with `ENSEMBLE_*` names.
- Added final-per-category, minimum-member-weight, and alignment-drop controls.
- Added floor method variants to `ENSEMBLE_METHODS`.

### Ensemble Builder Update

- Replaced holdout-fit routing with OOF-fit / holdout-eval comparison.
- Added production eligibility metadata and diagnostic-only stackers.
- Added category-floored candidate selection and alignment validation.
- Added 2% floor variants scored with their deployed post-floor weights.
- Added `build_production_ensemble()` and config provenance fields.

### Price Pipeline Update

- Replaced direct bakeoff/conformal code with `build_production_ensemble()`.
- Renamed MLflow dataset/tag values to `ensemble_selection`.
- Logged comparison, candidate metrics, and alignment metadata.

### Retrain Update

- Replaced broken `_recompute_ensemble` path with `_build_retrain_ensemble()`.
- Added explicit `reweight` and `reselection` modes.
- Renamed degradation threshold usage to `ENSEMBLE_DEGRADATION_THRESHOLD`.

### CLI Update

- Added `energy-forecasting deploy retrain --mode {reweight,reselection}` wiring.

### Runtime Naming Sweep

- Changed API fallback ensemble method to `inverse_mae`.
- Updated SLSQP-specific inference/model-store comments to neutral ensemble wording.
- Removed stale bakeoff wording from the price pipeline.

### Test Update - Ensemble

- Updated ensemble tests for 15 registered methods and OOF-fit routing.
- Added diagnostic stacker and post-floor scoring assertions.

### Test Update - Defaults

- Updated API/publish test fixtures from `slsqp_optimized` to `inverse_mae`.

### Plan Document Update

- Added review clarifications for post-retrain category selection, eligibility metadata, intersection-loss validation, post-floor scoring, and config provenance.

### Canonical Docs Update

- Updated `docs/stage5_model_training.md` for OOF-fit ensemble construction and explicit retrain modes.
- Updated `docs/source_repo_guide.md` to mark EP blend.py as historical/reference construction.

### Test Update - Retrain Modes

- Added `tests/test_deploy_retrain.py` for reweight vs reselection routing.

### Test Update - Alignment and Floors

- Added direct alignment validation tests for duplicates, y_true mismatches, and tiny intersections.
- Added category-floor selection coverage for best OOF-MAE and OOF-RMSE.

### Verification

- `conda run -n energy-forecasting pytest tests/test_ensemble.py tests/test_deploy_model_store.py tests/test_deploy_price_inference.py tests/test_api.py tests/test_deploy_retrain.py -q` -> 64 passed, 12 sklearn/LightGBM feature-name warnings.
- `conda run -n energy-forecasting ruff check <changed files>` -> passed.
- Full `conda run -n energy-forecasting ruff check` still reports unrelated pre-existing import-order issues in files outside this change set (`data/sources.py`, `modeling/gen_load.py`, several scripts, and unrelated tests).
- Stale terminology sweep leaves only intentional historical mentions in this bug-fix plan/log and legitimate `slsqp_optimized` method-name references.
