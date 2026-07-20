# Forecast Feature Fix

This directory collects all planning, review, and execution docs for one initiative:
fixing a **data-leakage inference bug** in the day-ahead price models and the retrain
it triggered.

## What the bug was

The price models consumed `prog_*` features derived from SMARD `prognostizierte_*`
columns — D+1 forecasts published ~16:00–18:00 UTC on delivery day D. These are
**structurally unavailable at the 08:00 UTC inference time**, so the deployed model
was training on information it could never have in production (leakage), and severely
underperforming. The fix replaces them with source-neutral `forecast_*` columns built
from our own gen/load forecast artifacts (waterfall: own forecasts → SMARD → actuals),
with strict own-only coverage required for live D+1 inference.

## Reading order

1. **`forecast_fix.md`** — the leakage fix plan (root cause, the `forecast_*`
   contract, DST handling, guards). *Implemented.*
2. **`forecast_fix_comments.md`** — adversarial review comments on that plan.
3. **`forecast_fix_retrain_plan.md`** — cleanup + retrain plan: MLflow
   archive-then-prune, checkpointed re-tune, production-window handling. *Partially
   executed.*
4. **`forecast_fix_retrain_implementation_log.md`** — running log of the retrain
   implementation (kept by the executing session).
5. **`forecast_fix_coverage_remediation_plan.md`** — **current active plan.** The
   first retrain's holdout was corrupted by a 3-month coverage hole in the gen/load
   historical-forecasts (2026-04→06), inflating MAE to 15.77 (own-covered ≈ 12.5).
   This plan covers the gen/load backfill and the price re-evaluation that must
   follow before anything ships.

## Current status (2026-07-14)

Leakage fix landed; first price retrain complete **but not shippable** (holdout
contaminated by the coverage hole). Daily price step disabled. Next action: the
gen/load backfill in the remediation plan (§2 Phase G), then price re-eval (§3
Phase P). See that plan for the full checklist.

## Related, outside this dir

- `../mlflow_conventions.md` — run lifecycle / tagging referenced by the retrain plan.
- `../stage6_inference_api.md` — the Stage 6 assumption this fix superseded.
- `../../scripts/forecast_fix_*.py` — archive + dataset-audit helpers.
