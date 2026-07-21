# Stage 10: AI Narrative Forecast Story

A new page presenting the operational forecast (currently `deploy/index.html`'s tab dashboard) in scrollytelling form, in the spirit of Stage 9 (`deploy/story/`), with AI-generated narrative text explaining recent price/generation/load conditions and what's driving tomorrow's forecast. Distinct from Stage 9: Stage 9 tells the decade-long historical story ("why does this market matter"); Stage 10 tells the forecast story ("what is the model saying, and why"). The two stay as separate pages for now, cross-linked, with a merge revisited later once both exist.

**Detailed plan:** this document
**Status:** implemented 2026-07-09, committed and deployed 2026-07-09. All of 10a-10f built and tested end-to-end against real production models and a real Groq call (see below). `GROQ_API_KEY` added as a GitHub Actions repo secret. The open question of how `deploy/story/forecast/data/*.json` (and Stage 9's `deploy/story/data/*.json`) stays fresh is resolved by `.github/workflows/story_data.yml` — a weekly (Monday 06:00 UTC, also `workflow_dispatch`-able) job that reruns `build_story_data.py` + `build_forecast_story_data.py` + `deploy narrative-yearly` and commits the refreshed JSON/inlined-HTML straight to `master`; since `deploy/story/` is git-tracked, the daily Pages deploy picks up the latest commit automatically with no separate restore step. Pages/repo visibility stays public (explicit user decision — not worth the friction of a private repo for a solo project at this stage).

Implementation notes vs. the plan below:
- SHAP computation had to account for a fact not known when this plan was written: each production price model is a MAPIE `CrossConformalRegressor` wrapping 5 cross-validation fold estimators, not a plain sklearn model. `model.predict(X)` is a *weighted* mean across those 5 folds (weights derived from `estimator_.k_`, not a flat 1/5). SHAP values and base values are computed per fold estimator and combined with those same weights — see `energy_forecasting/deploy/shap_attribution.py`. Verified against real production models: `base_value + Σ shap[feature]` reconstructs the actual ensemble `y_pred` to ~1.7e-5 EUR/MWh.
- SHAP feature matrices are keyed by model name, not feature_version, in case a future retrain has two production models sharing a feature_version with different scaler instances (not true today, but would silently misattribute if it became true).
- `deploy/data/price_shap.json` only publishes category-level attribution (base value + per-category signed contribution), not the raw per-feature matrix — matches the plan's schema.
- 34 new/updated tests added (`test_deploy_shap_attribution.py`, `test_deploy_narrative.py`, plus additions to `test_deploy_publish.py`); full suite is 535 passing, 0 regressions.
- Weekly yearly-recap script and CLI narrative commands are wired into the `Makefile` (`story-forecast-data`, `narrative`) but, matching Stage 9's own precedent, are **not** wired into any scheduled GitHub Actions workflow — manual invocation only.

---

## Context

The operational dashboard (`deploy/index.html`) presents forecasts as raw charts with no explanation — a reader has to already understand day-ahead electricity markets to get value from it. The user wants to eventually replace it with a narrative presentation that explains what the numbers mean, both for recent history and for tomorrow's forecast, before combining it with Stage 9's historical material into one final site.

Data check performed before writing this plan:
- `data/processed/merged.parquet` has full generation-by-fuel-type, load, and per-neighbour cross-border import/export columns, hourly, 2014-12-31 → present — multi-year comparisons are cheap, no new data collection needed for the yearly-recap sections.
- Stage 9's `scripts/build_story_data.py` already has reusable patterns: `GENERATION_GROUPS`/`RENEWABLE_GROUPS` categorization, NaN-safe JSON helpers (`nanlist`), and monthly/calendar-year aggregates (`price_history.json`, `generation_mix.json`, `negative_prices.json` — the last is per-calendar-year, not rolling, so Stage 10 needs its own rolling-window cut of the same logic).
- `energy_forecasting/deploy/price_inference.py` (`run_price_inference()`, lines 258-286) already loads each of the 5 production price models and builds their exact D+1 feature matrices (`X_d1`) in memory, and already persists per-day feature values into `feature_audit["feature_values"]`. `shap>=0.46` is already a `pyproject.toml` dependency. This means real per-instance SHAP attribution is available almost for free — no new dependency, and the expensive-looking part (model loading, feature engineering) already happens exactly where it's needed.
- `models/price_feature_cols.json` has 3 static ranked feature-set lists per model type (`fs_rfecv_optimum`, `fs_shap_top90`, `fs_shap_top247`) — used only as a fallback reference now, since real per-instance SHAP (below) supersedes static rankings as the driver-explanation source.
- `deploy/data/model_metadata.json` has the 5 production models + ensemble weights (written by `write_model_metadata()` in `energy_forecasting/deploy/publish.py`).
- `deploy_app` Typer group + `forecast_cmd` pattern confirmed at `energy_forecasting/cli.py:993, 1068-1082`.
- `.github/workflows/daily_forecast.yml` — single `inference` job, secrets passed via per-step `env:` blocks, only `GITHUB_TOKEN` used today.
- `deploy/data/gen_load_actuals.json` does **not** exist in the local working copy (only produced/cached in CI) — local testing of gen/load headline signals needs a real snapshot (`gh run download` / `gh release download`) before this can be exercised end to end locally.

---

## Decisions

- **Two separate cadences, two separate Groq calls.** Sections 2-3 (yearly recap) are slow-moving — recomputing them daily is wasted work, so they run on a **weekly** cadence via a periodic script (like Stage 9's `make story-data`). Sections 4-5 (forecast) run **daily**, tied to tomorrow's forecast. Each cadence gets its own Groq call producing multiple narrative fields in one structured JSON response.
- **"Last year" = rolling trailing 365 days**, not a fixed calendar year — always current, no year-end cutover needed.
- **Real per-instance SHAP, not static rank-weighted categories**, for the forecast driver explanation (see Stage 10c). This was a late but material change from the original design: computationally near-free (all 5 production models are tree-based or linear — the two cases SHAP handles exactly and fast) and only a modest engineering lift, since the model-loading and feature-matrix code already exists in `run_price_inference()`.
- **SHAP computation lives in the numeric pipeline (`price_inference.py`), not in the narrative module.** It's cheap, deterministic, and network-free, so it belongs in the always-runs path — it rides along with `feature_audit` the same way that's already published today.
- **HTTP via `requests`** (already a dependency, already the convention in `data/smard.py`/`data/commodities.py`) — no new SDK for a couple of POSTs/day.
- **Secret: `GROQ_API_KEY`**, read via `os.environ.get(...)` inline, warn-and-skip if absent — same convention as `FRED_API_KEY` (`energy_forecasting/data/commodities.py:306-309`).
- **Facts assembly is fully decoupled from the LLM call** — always runs, never depends on network; only prose generation degrades independently on Groq failure. Facts payloads are always persisted, even if the LLM call fails.
- **Narrative generation is its own CLI command**, separate from `deploy forecast` (its own CI step, its own `continue-on-error: true`) — a crash here must be physically incapable of touching the numeric pipeline.
- **New page, separate from Stage 9 for now**: `deploy/story/forecast/index.html`, cross-linked with Stage 9's closing CTA. Merge into one site revisited later.
- **Frontend loads live, not inlined.** Unlike Stage 9's data-island approach (needed for `file://` compatibility on static historical data), this page's data changes regularly and is always served over https in production, so charts/narrative load via plain `fetch()` of the published JSON.

---

## 1. Section content (user-authored, for reference)

1. **Introduction.** Framing paragraph on why forecasting matters for grid flexibility as renewables share grows — copy owned by the user, not regenerated here.
2. **Last year's generation and load.** Stacked area chart, daily generation by fuel type (reuse `GENERATION_GROUPS`), load as an overlaid line. Explanatory text: % generation by fuel type over the trailing year, imports/exports as % of domestic generation, AI summary of the year's trend vs. the prior year and the last 7 days vs. the same week ~52 weeks back.
3. **Last year's prices.** Hourly price chart over the trailing year. Explanatory text: mean price, most/least expensive hours-of-day on average, negative-price-hour count, AI summary of trends.
4. **Generation and load forecast.** AI-generated comparison of the forecast to recent history, highlighting anything extreme.
5. **Price forecast.** AI-generated comparison to recent history plus the real SHAP-based driver explanation (see 10c).

## 2. Stage 10a — Yearly recap facts pipeline

New script (sibling to `scripts/build_story_data.py`, reusing its `GENERATION_GROUPS`/`nanlist` patterns) reading `data/processed/merged.parquet`, computing over a **rolling trailing-365-day window**:
- Daily gen-by-fuel-group series + daily load line.
- Fuel-type mix %, imports/exports as % of domestic generation (`cross-border_flows_*_imports/exports` per neighbour).
- Hourly price series + stats: mean, most/least expensive hours-of-day on average, negative-price-hour count for the trailing year (rolling variant of the calendar-year logic behind `negative_prices.json`).
- The same block again for the trailing year *prior* to the current window (YoY comparison), plus last-7-days vs. the equivalent 7 days ~52 weeks back.

Runs periodically (weekly), via a new `make` target — not part of the daily pipeline.

## 3. Stage 10b — Yearly recap narrative (Groq call #1, weekly)

New function in `energy_forecasting/deploy/narrative.py` consuming 10a's facts. One Groq call, `response_format: json_object`, producing 2 fields: `gen_load_yearly_summary`, `price_yearly_summary`. Same failure contract as 10c: never raises, always persists facts even if the LLM call fails. Chained into the same weekly script as 10a, writing `deploy/story/forecast/data/narrative_yearly.json`.

## 4. Stage 10c — Forecast driver facts + narrative (Groq call #2, daily)

### SHAP computation (in `price_inference.py`, not `narrative.py`)

Inside `run_price_inference()`'s per-model loop (`price_inference.py:258-286`), right after each model's `.predict()` call:

- **Tree models** (LGBM, XGB×2, CatBoost): `shap.TreeExplainer(model).shap_values(X_to_predict)` — exact, tree-path-dependent baseline, no background sample needed.
- **Ridge**: closed-form linear contribution, `coef * (x - mean)`, in the already-scaled space `scaler.transform(X_d1)` produces.
- **Combine into one ensemble-level attribution** using the same weighted sum already applied to predictions: `ensemble_shap[feature] = Σ weight_m * shap_m[feature]`, aligned by column name across the union of the 5 models' feature-sets (zero-filled where a feature isn't in a given model's set). `ensemble_base_value = Σ weight_m * explainer_m.expected_value`.
- **Invariant to test**: `ensemble_base_value + Σ ensemble_shap[feature] ≈ y_pred` per hour, within floating-point tolerance.
- **Bucket into fixed categories** (gas / carbon / oil / neighbour-prices / wind / solar / residual-gen / price-momentum / calendar / load) via keyword match on column name, summing **signed** contributions per bucket per hour.
- Attach to `result.attrs["shap_attribution"]` alongside the existing `model_predictions`/`feature_audit`. Persisted by `publish.py` into a new `deploy/data/price_shap.json` (24 hours × per-category signed EUR/MWh contribution + base value). Runs on every `deploy forecast` invocation — cheap, deterministic, no network, not gated behind the narrative CLI.

### Facts payload (`narrative.py`)

Reads already-published JSON only, no model loading or feature engineering of its own:
- `deploy/data/price_forecast.json`, `actuals.json`, `gen_load/*_national.json`, `gen_load_actuals.json`, `model_metadata.json`, `errors_summary.json`, `price_shap.json` (new).
- Builds `category_attributions`: per category, signed mean contribution (EUR/MWh) and % of total absolute contribution for tomorrow's forecast, ranked by magnitude — read directly from `price_shap.json`, no recomputation.
- The originally-planned `headline_drivers` z-score-anomaly mechanism is dropped for driver explanation — real SHAP already says which real inputs are pushing tomorrow's forecast away from the model's own baseline, and by how much.

### Groq call

`POST https://api.groq.com/openai/v1/chat/completions`, Bearer `GROQ_API_KEY`, model `llama-3.3-70b-versatile` (CLI-overridable), `response_format: {"type": "json_object"}`, `temperature: 0.4`, `max_tokens: 700`, 20s timeout, single attempt, no retries. System prompt: state the model's own attributions factually ("category X contributed +N EUR/MWh vs. the model's baseline for this forecast"), with one explicit sentence of epistemic humility — this explains the model's reasoning, not verified real-world market causality. Strict JSON output, keys: `gen_load_forecast_note`, `price_driver_explanation` (plus whatever additional keys 10d's sections need once its structure is finalized).

Any failure (HTTP error, timeout, malformed JSON, missing keys) → treated uniformly as unavailable, logged as a warning, never raised.

### Output

`deploy/data/narrative_forecast.json` — `generated_at`, `delivery_date`, `model`, `status` (`ok`/`unavailable`), `reason` (`null`/`no_api_key`/`api_error`/`facts_error`), the narrative strings, and the full facts object (always written).

### CLI command

```python
@deploy_app.command("narrative")
def narrative_cmd(model: str = typer.Option("llama-3.3-70b-versatile")):
    """Generate the AI narrative for the daily forecast (non-blocking; degrades gracefully)."""
    from energy_forecasting.deploy.narrative import generate_narrative
    result = generate_narrative(model=model)
    logger.info(f"Narrative status: {result['status']}" + (f" ({result['reason']})" if result.get("reason") else ""))
```

### Workflow wiring

New step in `.github/workflows/daily_forecast.yml`'s `inference` job, after "Run inference (skip data update)" and before "Save deploy state":

```yaml
- name: Generate AI narrative (non-blocking)
  continue-on-error: true
  run: conda run -n energy-forecasting energy-forecasting deploy narrative
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

Requires the user to add `GROQ_API_KEY` as a GitHub Actions repo secret (manual step) — until then it silently writes `status="unavailable", reason="no_api_key"`.

## 5. Stage 10d — Frontend build

`deploy/story/forecast/index.html`, 5 sections per §1, built on Stage 9's shell (`story.css` tokens; a `common.js` extracted from `deploy/story/charts.js` for shared Plotly helpers). Charts and narrative load via `fetch()` of `deploy/story/forecast/data/narrative_yearly.json` (weekly), `deploy/data/narrative_forecast.json` + `price_shap.json` (daily), and the existing `deploy/data/*.json` forecast files.

## 6. Stage 10e — Cross-linking

Link from Stage 9's closing "Why forecasting matters" CTA to this new page. Leave the operational dashboard live in the meantime.

## 7. Stage 10f — Docs

Update this document's status line as each stage completes; update `docs/master_plan.md`'s status line.

---

## Build order

1. **Groq spike** (no repo code) — confirm `GROQ_API_KEY` + chosen model + `response_format: json_object` work against Groq's endpoint via a one-off script. Shared by 10b and 10c.
2. **10a** — yearly recap facts pipeline.
3. **10b** — yearly recap narrative.
4. **SHAP computation** — add to `run_price_inference()`, with the invariant unit test, before 10c's facts assembly is built.
5. **10c** — forecast driver facts + narrative, CLI command.
6. **Milestone**: review both narrative outputs (10b, 10c) against several real days/weeks for factual accuracy and tone before touching frontend.
7. **Test against real local data** — pull a `deploy/data/` snapshot via `gh run download`/`gh release download` for the missing `gen_load_actuals.json`.
8. **10d** — frontend build.
9. **10e** — cross-linking.
10. **CI wiring** — add `GROQ_API_KEY` secret, add the workflow step, trigger `workflow_dispatch`, confirm `narrative_forecast.json`/`price_shap.json` survive the cache/artifact round-trip and appear in the Pages artifact.
11. **10f** — docs.

## Verification

- Unit test: `ensemble_base_value + Σ ensemble_shap[feature] ≈ y_pred` per hour, for a fixed test day.
- Manually inspect `narrative_yearly.json` and `narrative_forecast.json` for 2-3 different real days/weeks: confirm no invented numbers, confirm driver language matches actual SHAP signs/magnitudes, confirm graceful `unavailable` behavior when `GROQ_API_KEY` is unset or a bad key is supplied.
- Trigger `workflow_dispatch` on `daily_forecast.yml`, confirm the new step's log shows `status=ok` (or a clean `unavailable` reason), confirm both new JSON files appear in the Pages artifact.
