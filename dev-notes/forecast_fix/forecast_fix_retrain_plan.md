# Forecast Fix — Cleanup & Retrain Plan

**Date:** 2026-07-13
**Depends on:** `forecast_fix.md` (implementation landed in commits `df9206c`, `121621d`)
**Status:** Decisions made — MLflow **staged archive-then-prune**, tuning **full re-tune with checkpoints**, production **disable price step** during the window. See §2, §5, §9.

---

## 0. Why this is needed

The forecast-feature fix replaced the price models' `prog_*` (raw SMARD, leaky at
inference) features with source-neutral `forecast_*` columns built from own
gen/load forecast artifacts. This changes the price feature **distribution** for
all 2022+ training rows and the feature **lists** themselves. Consequences:

- Every price model currently in `models/price/` was trained on leaky features and
  is invalid for production.
- The runtime guard (`_validate_trained_price_schema`) now **hard-fails**
  `run_price_inference` because the deployed `price_feature_cols.json` and price
  dataset schemas still contain `prog_*` tokens (verified: 13/15/27 `prog_` cols
  across the three feature versions). **Until this retrain completes and a new
  release ships, the daily price forecast errors rather than merely
  underperforming.**
- All previous price MLflow runs are subtly non-comparable to the new contract
  (their metrics were computed on leaky features).

---

## 1. Scope — what is and isn't invalidated

| Asset | Status | Reason |
|---|---|---|
| `price/feature_selection` (71 runs) | **Invalid** | Feature lists contained `prog_*`; importances/selections are on leaky features |
| `price/model_training` (5000 runs) | **Invalid** | Hyperparam search + metrics on leaky datasets |
| `price/production` (5 runs) | **Superseded** | Deployed ensemble; the 11.148 MAE baseline was leakage-inflated |
| `generation/*` (wind_onshore 83, wind_offshore 30, solar 61, load 61, gen_load_diff 14) | **VALID — keep, do not retrain** | Gen/load models were not touched. They produce the `historical_forecasts` artifacts the price builder consumes; those artifacts are unchanged. |
| `data/processed/historical_forecasts/*.parquet` | **Valid — keep** | Unchanged; consumed as-is by `build_forecast_columns` |
| `models/price/*.joblib`, `models/price_feature_cols.json` | **Replace** | Trained on leaky features |
| `data/processed/datasets/price_*.parquet` | **Regenerate** | Built from `prog_*`; must be rebuilt via `build_forecast_columns` |
| Deployed GitHub Release `models.tar.gz` | **Replace after backtest** | Contains old price models |

**Key scoping win:** this is a *price-only* retrain. The expensive gen/load
retrain (48 base + 16 ensembles, 8–12h) is **not** required.

---

## 2. MLflow disposition — DECISION REQUIRED

The store is 5.1 GB, dominated by `price/model_training` (5000 runs of model
binaries). The generation experiments stay as-is regardless. The decision is only
about the three **price** experiments. Per `docs/mlflow_conventions.md` the run
lifecycle is Active → Archived (`archived=true` + reason, still queryable) →
Deleted (reserved for "broken / wrong data").

Whichever option is chosen, do these **preservation steps first** (cheap, decouple
the "before" story from the 5.1 GB store):

1. `export_experiment_summary()` for `price/production` and the top
   `price/model_training` runs → `docs/archive/price_pre_forecast_contract/`.
2. Dump the tuned hyperparameters of the current production models to a versioned
   JSON in that archive dir (so they're reusable as tuning warm-starts without the
   MLflow store).
3. Record the leakage-inflated 11.148 MAE baseline explicitly as **not** a target
   to beat (see §6).

### DECIDED: staged archive-then-prune (Option A now → Option B after validation)

Two phases, deliberately not a one-shot, so nothing bulky is destroyed until the
new models are proven:

**Phase 2a — Archive in place (before/during retrain).**
Do the preservation steps, then tag all price runs `archived=true`,
`archive_reason="pre-forecast-contract; leaky/non-comparable"`,
`feature_contract=prog_leaky`. **Delete nothing yet.** This keeps the full 5.1 GB
and complete in-MLflow reproducibility while the retrain is validated. If the
retrain has to be compared against or reproduced from the old runs, everything is
still there.

**Phase 2b — Prune bulk (only after the new ensemble passes backtest and ships).**
Once §6 backtest is signed off and §7 release is out, delete the **artifacts**
(model binaries) of the superseded `price/model_training` runs to reclaim storage,
keeping their params/metrics metadata (tiny, in `mlflow.db`) and the
`price/production` + `price/feature_selection` runs for audit. Use `mlflow gc` /
`cleanup_orphaned_artifacts()`.

Rationale for staging: the recommendation was originally a straight archive+prune,
but pruning before the retrain is validated would discard the only fallback if the
new models regress or the run needs to be reproduced. Archive first, prune once
confident. Generation experiments are untouched in both phases.

> Not chosen: hard-deleting the price experiments outright (fastest storage
> reclaim) — rejected because it removes the in-MLflow "before" before we know the
> "after" is good.

### Contract-boundary tagging (all options)

Introduce a `feature_contract` tag on every run going forward:
`feature_contract=forecast_v1`. This is the durable comparability boundary — even
if old runs are deleted, new runs are self-describing. The required-tag plumbing
(`TrackedRun` validation + `mlflow_conventions.md` + all call sites) is a **Phase 0
code change** — see §3 item 4 — because it gates every subsequent training run.

> Note: per the conventions, a feature-set/data change is normally a "new
> experiment" trigger. We deliberately reuse the existing `price/*` experiment
> names (not `price/feature_selection_v2`) and rely on the `feature_contract` tag
> + archiving to separate epochs, to avoid experiment sprawl. The tag is the
> source of truth for comparability.

---

## 3. Phase 0 — code cleanup before retraining

Land these first so the retrain runs on clean code (from the implementation
review):

1. **Dead loader.** `load_gen_load_forecasts`, `FORECAST_TARGETS`,
   `_resolve_base_columns`, `_all_columns` in `modeling/gen_load_forecasts.py` are
   no longer used in production (only tests). Either delete them or refactor
   `build_forecast_columns` to reuse `load_gen_load_forecasts`. Keep `_align_tz`
   (still imported by the builder).
2. **Forked DST logic.** `_normalize_local_delivery_grid` in `forecast_inputs.py`
   re-implements `normalize_dst`'s spring-forward interpolation for tz-naive
   artifacts. It only fires for naive inputs (production artifacts are UTC), so it
   is low-risk, but it duplicates intent the plan said to avoid. Consolidate on
   `normalize_dst` or document why the naive path is separate.
3. **Confirm** `SHORT_NAMES` retains `prog_*` aliases for audit/parse only (already
   intended) and that no production feature list references them.
4. **`feature_contract` plumbing (code change, not docs).** `_REQUIRED_TAGS` in
   `modeling/mlflow_utils.py:30` is currently `{"stage", "feature_version"}`. Adding
   `feature_contract` there makes `TrackedRun.__enter__` raise for *every* call site
   that omits it — feature-selection, tuning, final training, and the production
   bakeoff. So this must land in Phase 0 as a coordinated change: add the required
   tag **and** pass `feature_contract=forecast_v1` at all call sites, with test
   coverage, before any retrain run. Update `docs/mlflow_conventions.md`'s
   required-tags list in the same change. Doing this last (as an afterthought) would
   either leave the retrain runs untagged or crash them on entry.

Each cleanup should keep the suite green (`pytest tests/`, currently 546 passing).

---

## 4. Phase 1 — regenerate datasets & sanity checks

1. **(Optional) refresh data.** `energy-forecasting update` (or `process`) only if
   the underlying SMARD/weather data is stale. `merged.parquet` itself is unchanged
   by this fix — the `forecast_*` columns are built during dataset prep, not merge —
   so this step is about data freshness, not the contract.
2. **Delete stale cached datasets first (mandatory).** `prepare_price_dataset`
   short-circuits on `if existing and not force: return existing` (`price.py:92`)
   **before** it validates the feature list or rebuilds `forecast_*` columns. A
   leftover `price_*.parquet` from the leaky epoch would therefore be reused
   silently — the retrain would train on old `prog_*` data while every downstream
   token check still passes (they inspect the clean *feature list*, not the stale
   *parquet*). Remove them explicitly before regenerating:
   ```
   rm -f data/processed/datasets/price_*.parquet
   ```
   (Alternatively add/plumb a `--force` flag through the `train price` CLI to
   `prepare_price_dataset(force=True)`; the `rm` is the zero-code path.)
3. **Regenerate price datasets** with the new contract (this happens automatically
   inside `train price` via `prepare_price_dataset` → `build_forecast_columns`, but
   run once standalone first for the parity audit):
   - Confirm 0 NaN across the eight `forecast_*` columns (verified on current data:
     0 NaN over 100,824 rows; source mix own=36,842 / smard=63,574 / actual=408 /
     missing=0).
4. **Parity audit** (from `forecast_fix.md` retrain step 3):
   - Report the own/smard/actual provenance mix per year (from
     `forecast_source_counts`).
   - Plot `forecast_gen_total` across the ~2022 own-vs-SMARD construction boundary
     to confirm the known structural break is not injecting spurious signal.
   - Spot-check 2022+ rows: `forecast_residual_load` = own load − own wind/pv, and
     differs from the old leaky `prog_residual`.

---

## 5. Phase 2 — retrain (research run): full re-tune, checkpointed

**Budget: full re-tune.** The old hyperparameters were optimized for leaky features
and the most important feature (`residual`) now carries proper diurnal signal it
previously lacked, so a genuine search is warranted. Feature *re-selection is
mandatory regardless* — the lists now reference `forecast_*` and importances will
shift, so `fs_rfecv_optimum` / `fs_shap_top*` must be recomputed. The long wall
time (RFECV ~1–2h + tuning across feature sets × model families) is de-risked by
the checkpointing below.

### Checkpointing & resumability (requirement)

A mid-run failure must **not** invalidate hours of completed work. The pipeline
already provides most of this; the plan is to run it in stages that align with the
existing checkpoint boundaries so a crash resumes at the last completed stage.

Built-in mechanisms (verified in code):
- **Trial-level (Optuna):** each tuning study is a per-study SQLite DB via
  `RDBStorage` with a deterministic `study_name` and `load_if_exists=True`
  (`tuning.py:128-144`). GridSampler(seed=0) means re-running the *same* command
  skips already-evaluated grid points — completed trials survive a crash.
- **Dataset-level:** `prepare_price_dataset` reuses cached
  `data/processed/datasets/price_*.parquet` (`price.py:93`), so datasets are not
  recomputed on re-run. This caching only helps *after* §4 step 2 has deleted the
  leaky-epoch parquets — the cache then holds the freshly-regenerated clean
  datasets, and resuming a crashed tune reuses those rather than the old ones.
- **Model-level resilience:** per-model tuning is wrapped in try/except
  (`price.py:385, 396`), so one model family failing does not abort the run.

Gap to manage: **RFECV feature selection is not cached** — a crash *after* selection
but *before/within* tuning would otherwise redo the ~1–2h RFECV. Mitigate by
splitting selection from tuning:

1. **Stage A — feature selection (once).**
   ```
   energy-forecasting train price --feature-selection --use-rfecv --top-k 4
   ```
   Let this reach the point where it has written the `fs_rfecv_optimum` /
   `fs_shap_top*` datasets and logged the selected feature versions, then capture
   the winning `feature_version` name(s). (If it proceeds straight into tuning,
   that is fine — the Optuna storage makes the tuning resumable anyway.)
2. **Stage B — pinned tuning (resumable).**
   ```
   energy-forecasting train price --pin-feature-version fs_rfecv_optimum
   ```
   Pinning skips re-running RFECV entirely and tunes only the chosen dataset. If it
   dies, re-run the identical command — cached datasets + Optuna SQLite resume
   completed trials; only the incomplete study continues.

Operational requirements:
- Launch **detached** per `~/.claude/CLAUDE.md` (setsid nohup, save PID to
  `<log>.pid`); consider `--parallel N` for the tower's 16 cores.
- Before launch, snapshot `OPTUNA_DIR` and the datasets dir so a resume starts from
  a known-good checkpoint; on failure, inspect the log, fix, and re-run the same
  command (do **not** delete the Optuna DBs — that discards the checkpoints).
- Tag every run `feature_contract=forecast_v1` + the standard
  `stage`/`feature_version`/`holdout_days`/`cv_*` tags.

Subsequent steps (handled by the pipeline, but verify each):
- Re-blend ensemble weights (SLSQP on holdout).
- Re-calibrate conformal PI on holdout residuals → `conformal_quantile` in config.

> Optional hardening (only if a stage-boundary checkpoint proves insufficient):
> cache the RFECV result to disk keyed by dataset hash + feature-list hash so even
> Stage A is resumable without pinning. Not required given the split above.

---

## 6. Phase 3 — backtest & honest-baseline framing

Evaluate the new ensemble on:
- High-solar days (summer midday) — where the old flat/stale `prog_residual`
  hurt most.
- Negative-price hours — where near-zero residual signal matters.
- Large residual-load ramps.

**Framing (critical to avoid a false "regression"):** the old 11.148 EUR/MWh
holdout MAE was measured with the leaky SMARD `prog_*` values present — a figure
never achievable in live production. The new holdout uses own-model `forecast_*`
(noisier than TSO operator forecasts), so an honest holdout MAE will likely be
numerically **worse** than 11.148, and that is the fix working. Record the honest
number and, if a like-for-like baseline is wanted, recompute the old pipeline's
holdout MAE using only inference-available inputs. The meaningful comparison is
production error before vs after, not new-holdout vs 11.148.

Also sanity-check that not every derived `forecast_*` helps: `forecast_gen_other`
and `forecast_gen_total` are noisier differences of model outputs — if feature
selection or an ablation says they hurt, drop them.

---

## 7. Phase 4 — release & re-enable production

1. `energy-forecasting deploy export-models` → fresh `models/price/`,
   `price_feature_cols.json` (now `forecast_*`, passes the runtime guard).
2. Archive the old `models/price_feature_cols.json` and price models under
   `docs/archive/price_pre_forecast_contract/` (or delete after backtest sign-off).
3. Rebuild `models.tar.gz`; publish a new GitHub Release.
4. **Production during the retrain window — DECIDED: disable the price step.**
   Because the runtime guard hard-fails inference on the old schema, temporarily
   disable the price step in `.github/workflows/daily_forecast.yml` (keep gen/load +
   dashboard running on last-good data) as the **first action** of this whole job,
   before archiving/retraining begins — not just at release time — so the daily job
   is never visibly erroring. Re-enable it once the new release ships. Track the
   disable/re-enable as an explicit commit pair so it is easy to see the window.
5. First post-release daily run: confirm 24/24 `forecast_*` complete, strict mode
   passes, SHAP driver attribution resolves `forecast_*` names, and the published
   price forecast is sane.

---

## 8. Storage reclamation (if Option B or C)

- After preservation exports, delete superseded artifacts and run
  `mlflow gc --backend-store-uri sqlite:///mlflow.db` to purge them from disk.
- Run `cleanup_orphaned_artifacts()` to catch model/dataset blobs no longer
  referenced by any active/archived run.
- Target: reclaim the bulk of the 5.1 GB (dominated by `price/model_training`
  binaries) while keeping generation experiments and price run metadata.

---

## 9. Decisions — CONFIRMED

1. **MLflow disposition** — staged: **archive in place now (2a)**, then **prune bulk
   after the retrain is validated and shipped (2b)**. Generation experiments
   untouched.
2. **Tuning budget** — **full re-tune**, run in checkpointed stages (§5) so a
   failure resumes rather than restarts.
3. **Production during the retrain window** — **disable the price step** in the daily
   workflow up front; re-enable at release.

---

## 10. Review comments

These comments come from a follow-up review of the plan against the implemented
forecast-feature code and current CLI behaviour. Three findings about dataset-cache
reuse, `feature_contract` plumbing, and its ordering have been folded into §3, §4,
and §11; the execution-time reminders below remain.

1. **The production-disable step needs exact workflow semantics.**
   Define whether disabling the price step means skipping `run_price_inference`,
   publishing gen/load only, leaving the previous price JSON untouched, marking
   price status stale/unavailable, and/or skipping narrative/SHAP steps that depend
   on price output.

2. **Distinguish MLflow display paths from registry keys.**
   The prose correctly uses paths such as `price/model_training`, but helper calls
   often expect registry keys such as `price_model_training` and `price_production`.
   Any archive/export commands should state which identifier they expect.

3. **Storage cleanup should be dry-run and artifact-specific.**
   Before pruning, produce a manifest of run IDs and artifact paths, back up the
   archive, then delete only model binaries for archived `price_model_training`
   runs. Keep old `price_production` artifacts until the new release has completed
   at least one clean daily run.

4. **Add phase-level acceptance gates.**
   Useful gates include: regenerated dataset schemas contain zero banned tokens;
   `price_feature_cols.json` contains zero banned tokens; `run_price_inference`
   passes strict forecast construction on a test date; SHAP driver names resolve to
   `forecast_*`; the first post-release workflow publishes all expected JSON files.

5. **Add explicit "do not proceed" checks before full tuning.**
   Stop before the expensive run if regenerated datasets still contain `prog_`, if
   `forecast_source_counts` is missing from the parity audit, or if `actual`
   fallback appears in the holdout window.

---

## 11. Sequenced checklist

- [x] §7.4 disable price step in `daily_forecast.yml` (FIRST — before anything else)
- [x] §3 Phase 0 code cleanup (dead loader, forked DST), suite green
- [x] §3.4 `feature_contract` plumbing: add to `_REQUIRED_TAGS` + all call sites + `mlflow_conventions.md` + tests (BEFORE any retrain run)
- [x] §2 preservation exports + hyperparameter dump → `docs/archive/price_pre_forecast_contract/`
- [x] §2a archive-in-place: tag price runs `archived=true` + `feature_contract=prog_leaky` (delete nothing)
- [x] §4.2 delete stale `data/processed/datasets/price_*.parquet` (prevents silent reuse of leaky datasets)
- [x] §4 regenerate datasets + parity audit + provenance mix + boundary plot
- [x] §5 Stage A: feature selection (`--feature-selection --use-rfecv`), capture winning feature_version
- [ ] §5 Stage B: pinned tuning (`--pin-feature-version …`), detached + PID saved, `feature_contract=forecast_v1`
- [ ] §5 verify ensemble reblend + conformal recalibration — Stage A produced an honest `forecast_v1` ensemble at MAE 15.772; not promoted pending Stage B/backtest
- [ ] §6 backtest (high-solar / negative-price / ramps) + honest-baseline write-up
- [ ] §7 export models, rebuild `models.tar.gz`, GitHub release
- [ ] §7 re-enable price step in daily workflow; verify first clean run
- [ ] §2b prune bulk (`mlflow gc` on superseded `model_training` artifacts) — ONLY after release signed off
- [ ] update `CLAUDE.md` status
