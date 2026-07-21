# Merge Plan Design Decisions

Key architectural decisions made during comparison of `EP_EMA_merge_plan.md` (Plan A) and `docs/merge_evaluation.md` (Plan B).

**Date:** 2026-03-26

---

## 1. Feature Engineering Architecture

**Decision:** Dual system (DSL for market features, config dict for weather features), with a single flat feature list as the user-facing interface.

Price model feature selection informed by load model Optuna results — since Optuna is infeasible for the full price search space, use features selected by load models as starting candidates for price models.

**Why:** Load and price share many relevant features (weather, generation patterns, temporal signals). This leverages Optuna's selection without the computational cost of running it on the full price model. Weather FE involves boolean toggles and physics computations (compute air density? moist or dry? which spatial aggregation?) that don't map naturally to a suffix grammar, but the *outputs* of weather FE appear in the same flat feature list as market features.

**Sources:** Plan A (dual system architecture), with Plan B's single feature list ergonomics. Feature selection flow is new — neither plan had this.

## 2. MLflow Experiment Structure

**Decision:** Plan A's focused experiments with comparability rule at moderate granularity (~8-12 experiments).

The key rule: within a single experiment, runs must be comparable (same holdout, CV strategy, features — unless features are the thing being tested). Examples: `price/feature_selection`, `price/model_training`, `price/production`, `generation/wind_onshore`, `generation/solar`.

**Why:** Too few experiments (Plan B's one-per-target) risks the "overlapping mess" EP had. Too many (Plan A at full granularity) makes the MLflow UI its own navigation problem. Model type varies within an experiment and is captured in tags.

**Sources:** Plan A.

## 3. Ensembling

**Decision:** Compare inverse-MAE blend vs stacking as part of the train/retrain pipeline. Auto-select the best-performing method per target. Not a fixed architectural choice.

**Why:** Trivial to compare both on holdout data; make it data-driven rather than a design decision. Implement both, let the pipeline pick the winner per target at each retrain cycle.

**Sources:** Neither — both plans picked a fixed approach. This makes it empirical instead.

## 4. DuckDB

**Decision:** Deferred. Use Parquet files directly for now. Add DuckDB when needed (API, dashboard analytics).

Multiple Parquet files organized by source/domain (separate update cadences, schemas, granularity). When DuckDB is added later, define views over existing Parquet files — no migration needed, no schema required upfront.

**Why:** Minimum moving parts during the merge. DuckDB reads Parquet natively, so there's zero cost to deferring.

**Sources:** Plan A.

## 5. Dataset Tracking

**Decision:** Use MLflow's proper `mlflow.log_input()` dataset field. Training runs reference their datasets through MLflow's built-in dataset tracking, not the tag-based workaround EP used.

**Why:** EP tracked datasets as models (wrong abstraction) or via ad-hoc tags. MLflow has a first-class dataset concept — use it correctly.

**Sources:** Plan B (correct MLflow usage), rejecting Plan A's "no MLflow for datasets" simplification.

## 6. Scope of Initial Merge

**Decision:** Reproduce existing functionality first, with two exceptions:

- **Combined dashboard from the start.** It would make no sense to produce two separate dashboards and then merge them.
- **API from the start.** An API is already partially built in EP. Extend it rather than building a static-file architecture and then replacing it.

All other extensions (weather features directly in price model, multi-day forecasting, DuckDB, chatbot, quarter-hourly) are deferred until the merge is stable.

**Why:** The merge is already complex. But dashboard and API are user-facing deliverables that would be wasted work to build twice.

**Sources:** Plan A (reproduce-first discipline), with pragmatic exceptions.

## 7. Cleaning Rules Format

**Decision:** Python function calling small helper functions, not YAML or dataclasses.

```python
def clean(df):
    df = clip_bounds(df, "temperature_2m_*", min=-45, max=50)
    df = fill_zero_after(df, "stromerzeugung_kernenergie", after="last_valid")
    df = interpolate_gaps(df, method="cubicspline", max_gap=5)
    return df
```

**Why:** Dataclasses added abstraction (9 rule types, an interpreter, a type union) for no real benefit — the rules never need to be inspected programmatically. A function with commented helper calls is simpler, equally readable, and separates config (what/why, in `config/cleaning.py`) from logic (how, in `data/processing.py`).

**Sources:** Evolved from Plan B during review.

## 8. EP Tests

**Note:** Plan A (working from GitHub) stated EP has no tests. Plan B (working locally) found an 11-file pytest suite. The tests are gitignored — this explains the discrepancy. The tests exist and should be ported.

## 9. Price Model Tuning

**Decision:** Use Optuna with `GridSampler` for price models. Use Optuna with standard samplers (TPE) for generation/load models. One tuning API for everything.

**Why:** The GBT hyperparameter landscape for price models is smooth — grid search is sufficient. But since Optuna is already a dependency for EMA generation models, use it everywhere for consistency. `GridSampler` gives grid search semantics through the Optuna API.

**Sources:** Plan A (grid search reasoning), implemented via Optuna for unified tooling.

## 10. Time-Series Cross-Validation

**Decision:** Adopt EMA's `compute_timeseries_split_cutoffs` which ensures folds start/end at day boundaries, consistent train/test ratios, and forecast horizon matching the actual 24-hour prediction task.

**Why:** Day-boundary alignment matters for day-ahead forecasting. You're evaluating a coherent 24-hour block, not individual hours. EP's `TimeSeriesSplitter` doesn't enforce this.

**Sources:** Plan A.

## 11. Deployment Structure

**Decision:** Two workflows (daily forecast + periodic retrain). The daily workflow follows Plan A's ordering: parallel data collection → generation/load inference → price inference → deploy. If weather features go directly into price models, the gen/load → price ordering constraint relaxes.

**Why:** Two independent workflows are easier to reason about than a 4-step chain (EMA). Ordering within the daily workflow handles the dependency between gen/load and price forecasts.

**Sources:** Synthesis of both plans.

## 12. Feature Engineering: Optuna-to-Price Flow (Clarification of #1)

**Decision:** The Optuna search on generation/load models selects not just *which* weather features to include, but *how* they are computed (e.g., wet vs dry air density, spatial aggregation method, turbulence window size). These FE computation decisions transfer directly from load models to price models. Individual weather feature inclusion/exclusion in the price model is still handled separately during price feature selection.

**Why:** The physics computation choices (air density mode, aggregation strategy) are expensive to search over and the optimal choices are likely target-independent — wind power density computed the right way for generation forecasting is also computed the right way for price forecasting. Individual feature relevance (does the price model benefit from turbulence intensity?) is target-specific and handled in the DSL feature list.

**Sources:** New — extends decision #1.

## 13. Config Format Principle

**Decision:** Separate code from config, but use Python for config (not YAML/JSON) unless the config is pure data with no logic.

- **Cleaning rules** → Python dataclasses (they're parameterised actions, benefit from type checking/IDE support)
- **Availability/leakage rules** → Python dataclasses (same reasoning)
- **Feature lists** → Python lists of strings (the suffix DSL)
- **Short name registry** → Python dict
- **eu_locations** → JSON or YAML (pure data: coordinates, capacities, TSO mappings — no logic)
- **Hyperparameter search spaces** → Python dicts (Optuna API requires Python)

**Why:** The distinction is: does the config contain *logic* (actions, computations, conditionals) or *data* (coordinates, mappings, constants)? Logic benefits from Python tooling. Data benefits from language-independent formats. eu_locations is clearly data; cleaning rules are clearly logic.

**Sources:** Synthesis — resolves the apparent inconsistency between Python cleaning rules and JSON location data.

## 14. Model Storage

**Decision:** GitHub Releases, not git. Upload `.joblib` model files as release assets. Daily workflow downloads models for inference. Retrain workflow uploads new models.

**Why:** Keeps the repo clean. Models are ~100MB per retrain cycle — committing them to git means the repo grows unboundedly. Releases are versioned and downloadable. EP already uses this pattern for the merged dataset.

**Sources:** Plan A.

## 15. Prediction Interval Blending

**Decision:** Weighted average of individual intervals to start. Track prediction interval coverage as a metric from day one. Upgrade to calibrated blend intervals if coverage is consistently wrong (e.g., >97% when targeting 95%).

**Why:** Simple approach is quick to implement and may be good enough. Tracking coverage ensures we detect if it isn't. The monitoring creates a natural trigger for upgrading.

**Sources:** Plan A (simple approach), with Plan B's monitoring-driven upgrade path.

## 16. Dataset Caching

**Decision:** Hash-based filesystem cache. Hash the feature config (feature list + cleaning config + source data modification time). If a cache file exists with a matching hash, load it. Otherwise recompute and cache.

**Why:** Feature computation for the full pipeline will take minutes. Hash-based caching is ~20 lines of code, no MLflow dependency for the cache itself. Training runs still reference the dataset via `mlflow.log_input()` (decision #5) — caching and tracking are complementary.

**Sources:** Plan A (hash-based approach).

## 17. Python Version

**Decision:** Python 3.13.

**Why:** All critical dependencies work on 3.13 (tested 2026-03-27):
- MAPIE 0.9.2, CatBoost 1.2.10, XGBoost 3.2.0, LightGBM 4.6.0, scikit-learn 1.5.2
- MLflow 3.10.1, Optuna 4.8.0, FastAPI 0.135.2, pysolar, holidays, yfinance, fredapi
- MAPIE + each tree model (LightGBM/XGBoost/CatBoost) smoke tested with conformal prediction intervals

No reason to use an older version.

**Sources:** Tested locally.
