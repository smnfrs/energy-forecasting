# Forecast Fix Plan Review Comments

**Date:** 2026-07-13
**Scope:** Review comments on `docs/forecast_fix.md`; no implementation changes.

---

## Findings

### 1. DST guidance conflicts with the repo's current timestamp contract

`docs/forecast_fix.md` says spring-forward has 23 rows and fall-back has 25, and asks for 23/25-hour tests. But current `merged.parquet` is tz-naive and has 24 non-duplicate rows on the DST transition dates checked:

- `2024-03-31`
- `2024-10-27`
- `2025-03-30`
- `2025-10-26`

Stage 6 and validation also still assume 24 rows (`docs/stage6_inference_api.md`, `deploy/validation.py`).

Recommendation: either remove the 23/25 claim and state "validate against `strict_index`, currently 24 tz-naive delivery labels", or explicitly make DST handling a broader timestamp-model change touching inference, validation, publish/API/dashboard, and tests.

### 2. A stale trained feature schema could bypass the proposed guard

The plan focuses on `PRICE_FEATURES_*` and dataset prep, but inference loads trained columns from dataset/model metadata, then computes `PRICE_FEATURES_MAX` with validation disabled. If `price_feature_cols.json` or a saved dataset schema still contains `prog_*`, the config guard alone is not enough.

Recommendation: add an acceptance criterion and runtime/release check that trained feature schemas and exported `price_feature_cols.json` contain no `prog_` / `pct_prog_` tokens.

### 3. Actual fallback is still conceptually leaky unless it is tagged/audited

The doc allows actuals into model-ready `forecast_*` features for old rows. That may be acceptable as a pragmatic pre-coverage bridge, but it should be visible and measured.

Recommendation: add source/provenance counts to the parity audit: own forecast vs SMARD fallback vs actual fallback by year and feature. Either exclude actual-fallback rows from feature-selection sensitivity checks or report their impact.

### 4. Partial historical coverage needs a source-coherence rule

Strict D+1 mode checks all five artifacts, but historical waterfall rows could still mix sources if one artifact has a gap: own load plus SMARD total, or own wind/solar plus actual fallback. That can create artificial jumps in `forecast_gen_total`, `forecast_gen_other`, and `forecast_residual_load`.

Recommendation: add a builder invariant for historical rows: either require all five own artifacts for a row before using the own layer, or emit per-column/per-row `forecast_source_*` audit data and test partial-coverage behaviour.

### 5. The stale comments list should name exact active code comments

The plan says to fix comments generally, but there are known misleading active comments that should be listed explicitly:

- `energy_forecasting/config/columns.py`: forecast comment says `prog_*` values are "published for today - no lag required".
- `energy_forecasting/config/columns.py`: old comment says EMA forecasts overlay onto `prog_*` columns.
- `energy_forecasting/deploy/gen_load_inference.py`: `update_historical_forecasts` docstring still says files are used by the EMA overlay to supply `prog_*` features.
- `energy_forecasting/deploy/price_inference.py`: comments still describe applying the EMA overlay after D+1 extension.

Recommendation: list these in Step 6 / cleanup so they do not survive implementation.

---

## What Looks Good

The updated plan correctly fixes the earlier overstatement that "no guard exists": the real bug is the wrong `prog_*` availability rule. The `pct_forecast_*` producer issue is also now captured properly, and the warning not to delete blanket forward-fill wholesale is important and correct.
