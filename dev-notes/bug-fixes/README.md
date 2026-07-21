# Bug-Fix Plans

This directory indexes post-deployment fixes that are broader than a single
code patch and need their own design notes, verification plan, or retrain
sequence.

## Active and recent fixes

- [EP fidelity reproduction](ep_fidelity_reproduction_plan.md) - **current** price
  ensemble design. Reverts the bakeoff and reproduces EP verbatim: category floor
  (2 per model family) + inverse-MAE weights fit on the recent holdout, base models
  refit from fresh `merged` each retrain, bakeoff retained only as a standalone
  diagnostic (`scripts/ensemble_method_comparison.py`). Bootstrapped into production
  2026-07-20.
- [Price ensemble construction fix](ensemble_construction_fix.md) - **superseded**
  by the EP-fidelity plan above. The holdout-fit bakeoff / fair-OOS selection /
  no-refit design was a divergence from EP and was reverted.
  Implementation notes: [audit log](ensemble_construction_fix_implementation_log.md).
- [Forecast feature leakage](../forecast_fix/README.md) - the existing
  forecast-fix workstream for replacing unavailable `prog_*` features with
  source-neutral `forecast_*` features and repairing the downstream retrain.

## Sanctioned deviations from EP

The design principle for price forecasting is to **reproduce EP exactly** before
introducing improvements (master_plan.md decision #6). Any departure from EP must
be recorded here with its rationale. Everything not listed is expected to match EP.

| # | Deviation | Rationale | Status / trigger to revisit |
|---|---|---|---|
| 1 | **Conformal prediction intervals.** EP is point-forecast only; we add a symmetric conformal PI calibrated on the recent-holdout residuals of the deployed ensemble. | The dashboard, API, and story site all consume the interval. | Kept. Because the same holdout fits the inverse-MAE weights **and** calibrates the residuals, the PI is doubly in-sample and likely **materially too narrow**; monitor live-forward coverage as the trigger to graduate a separate calibration window. |
| 2 | **Candidate selection on OOF, weights on holdout.** The category floor (`select_final_models`) ranks best-MAE/best-RMSE per family on **OOF** predictions; inverse-MAE weights are then fit on the **recent holdout**. EP selects *and* weights on the recent holdout. | Selecting the floor out-of-sample is more robust than selecting on the same window the weights are fit to. | Intentional. Revisit only if it materially changes membership vs an EP-exact (holdout-selected) floor. |

`docs/forecast_fix/` remains in its existing location because it already
contains a live set of plans, review notes, JSON validation output, and
implementation logs. Treat it as part of this bug-fix index even though it is
not physically nested under this directory.
