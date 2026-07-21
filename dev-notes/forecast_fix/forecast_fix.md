# Forecast Feature Fix

**Date:** 2026-07-13  
**Status:** Approved for implementation

---

## Problem

The price model currently consumes feature names derived from raw SMARD source columns (`prognostizierte_*`). These columns are SMARD day-ahead generation/load forecasts published by German TSOs for delivery day D+1. The public forecast pipeline runs at **08:00 UTC on day D**. At that time, the D+1 SMARD forecast values are not available to the production forecast pipeline and must not be used for price inference.

This creates a training/inference mismatch. Historical `merged.parquet` contains SMARD D+1 values because the data was downloaded retrospectively, but those same values are structurally unavailable for the live 08:00 UTC forecast.

The upstream `energy_prices` work solved the same availability problem by ensuring delivery-day price features are backed by own generation/load forecast artifacts rather than by unavailable SMARD D+1 columns. The merged repo partially reproduced this idea through `_overlay_ema_forecasts`, but the implementation and documentation stayed source-named and incomplete. In particular, several important price-driver columns could still be raw SMARD values or flat forward-fills at inference time.

### Affected raw SMARD columns

These raw columns can remain in `merged.parquet` for audit and historical fallback. They must not be model-ready production price feature names.

| Raw SMARD column | Old short name | Replacement model-ready column |
|---|---|---|
| `prognostizierte_erzeugung_onshore` | `prog_gen_wind_on` | `forecast_gen_wind_on` |
| `prognostizierte_erzeugung_offshore` | `prog_gen_wind_off` | `forecast_gen_wind_off` |
| `prognostizierte_erzeugung_photovoltaik` | `prog_gen_solar` | `forecast_gen_solar` |
| `prognostizierter_verbrauch_gesamt` | `prog_load` | `forecast_load` |
| `prognostizierte_erzeugung_wind_und_photovoltaik` | `prog_gen_wind_pv` | `forecast_gen_wind_pv` |
| `prognostizierte_erzeugung_gesamt` | `prog_gen_total` | `forecast_gen_total` |
| `prognostizierte_erzeugung_sonstige` | `prog_gen_other` | `forecast_gen_other` |
| `prognostizierter_verbrauch_residuallast` | `prog_residual` | `forecast_residual_load` |

`prog_residual` is especially high risk because it is the most important price feature in the current `fs_rfecv_optimum` set. At live inference it can receive a stale or flat value instead of a proper diurnal forecast, which is most damaging around solar peak and negative-price hours.

---

## Why This Happened — Clarity Failures

### 1. Incorrect timing comments

`price_inference.py` and `docs/stage6_inference_api.md` state or imply that SMARD D+1 `prognostizierte_*` values are available to the 08:00 UTC forecast run. That is wrong. The corrected contract is: live D+1 price inference must use own gen/load forecasts, not SMARD D+1 values.

The daily workflow comment that says the run happens after D+1 SMARD generation forecasts publish is wrong for the same reason.

### 2. `_overlay_ema_forecasts` hid an incomplete contract

The name suggests a patch over SMARD columns, not the construction of a model-ready forecast feature set. It also left unclear which delivery-day quantities were covered. `forecast_gen_total`, `forecast_gen_other`, and `forecast_residual_load` are derivable from own forecast artifacts and should not be left as raw SMARD values.

### 3. Source names were used as feature names

`prog_residual`, `prog_gen_total`, and `prog_gen_other` sound like generic forecast features, but they encode a specific source with specific availability constraints. Price feature names should encode model availability semantics, not the raw source table they came from.

### 4. The leakage guard already exists — its availability rule is wrong

This is the true root cause, and it is **not** "no guard exists." The repo already has the EP-equivalent guard, and it is wired in:

- `AVAILABILITY_RULES` (`config/availability.py`) and `validate_features()` (`features/validation.py`) are the direct analogue of EP's availability rules / `validate_pipeline_leakage()`.
- It runs by default: `engineer_features` calls `validate_features()` when `validate=True` (`features/engine.py:188-189`), and there is a CLI validation path (`cli.py:824`).

The bug survived because of the rule at `config/availability.py:29-40`:

```python
AvailabilityRule("prog_*", 0, None, "TSO forecasts published for delivery day"),
AvailabilityRule("pct_prog_*", 0, None, "Derived from forecasts, same availability"),
```

`max_offset_days=0` means "available at inference time, no lag required." The rule **encodes the same false belief** as the timing comments in §1, so the validator passed `prog_residual`, `prog_gen_total`, and `prog_gen_other` because it was explicitly told they were safe. The guard did not miss the bug; the rule lied to it.

Implication for the fix: **amend the existing rule** (Step 7), do not bolt a second parallel prefix-blocklist beside it. Remove/repurpose the `prog_*` and `pct_prog_*` rules and add `forecast_*` / `pct_forecast_*` rules (legitimately `offset=0`, because own gen/load models produce them by 08:00 UTC). A future developer who adds `prog_foo` to a price feature list must then fail `validate_features()` from the single source of truth — two guards that can disagree is worse than one correct rule.

### 5. Forecast artifacts were named and documented ambiguously — and the residual was already computed but never consumed

Files under `data/processed/historical_forecasts/*.parquet` are not raw truth. They are forecast artifacts with `y_pred` values. Depending on how they were generated, rows may represent out-of-fold training forecasts, backtest forecasts, or live production forecasts. The price feature builder should consume them as forecast artifacts and should not describe them generically as EMA truth or SMARD replacements.

Sharper diagnosis of the gap: the residual data was never actually unavailable. `gen_load_forecasts.py:127-133` **already loads** `gen_load_diff` (its `FORECAST_TARGETS` line 29) and **already computes** `_derived_forecast_residual = load − wind_on − wind_off − solar` — which is exactly `forecast_residual_load`. The defect is that `_overlay_ema_forecasts` never *consumed* the residual (or `gen_load_diff`), not that the artifact was missing. `build_forecast_columns` should reuse the loader's existing derivation rather than rebuild it.

---

## Naming Contract

Use these namespaces consistently:

| Namespace | Meaning | Examples |
|---|---|---|
| Raw source columns | Columns copied from external data sources. Kept for audit and historical fallback. | `prognostizierte_*`, `stromerzeugung_*` |
| Model-ready forecast features | Best available forecast value under the defined availability rules. Price models consume these. | `forecast_load`, `forecast_residual_load` |
| Targets | Supervised learning labels. | `target_price`, `target_load` |
| Model outputs | Predictions emitted by trained models or forecast artifacts. | `y_pred`, `prediction_*` |

Avoid using `EMA` as a generic synonym for own forecasts. Use `own forecast`, `gen/load forecast`, `forecast artifact`, or `forecast waterfall`. Use `upstream EMA` only when referring historically to the source repo.

---

## Proposed Contract

Introduce source-neutral `forecast_*` columns. Price models consume only these generation/load forecast columns, never raw `prognostizierte_*` columns or old `prog_*` short names.

| New column | Meaning |
|---|---|
| `forecast_load` | Best available load forecast |
| `forecast_gen_wind_on` | Best available onshore wind forecast |
| `forecast_gen_wind_off` | Best available offshore wind forecast |
| `forecast_gen_solar` | Best available solar forecast |
| `forecast_gen_wind_pv` | Wind + offshore wind + solar forecast |
| `forecast_gen_total` | Total generation forecast |
| `forecast_gen_other` | Total generation forecast minus wind/PV forecast |
| `forecast_residual_load` | Load forecast minus wind/PV forecast |

### Forecast artifact store semantics

`data/processed/historical_forecasts/*.parquet` is a forecast artifact store. The builder consumes the `y_pred` column from these files:

```text
wind_onshore_DE_NATIONAL.parquet    -> forecast_gen_wind_on
wind_offshore_DE_NATIONAL.parquet   -> forecast_gen_wind_off
solar_DE_NATIONAL.parquet           -> forecast_gen_solar
load_DE_NATIONAL.parquet            -> forecast_load
gen_load_diff_DE_NATIONAL.parquet   -> gen_load_diff_forecast
```

If artifact metadata is added later, use an explicit origin field such as `oof`, `backtest`, or `production`. Do not make the builder infer semantics from filenames alone beyond the target/region mapping.

`gen_load_diff = total_generation - total_load` (positive means generation surplus). This sign convention is established in both source repos and in the merged repo's gen/load training code. Therefore:

```text
forecast_gen_wind_pv   = forecast_gen_wind_on + forecast_gen_wind_off + forecast_gen_solar
forecast_gen_total     = forecast_load + gen_load_diff_forecast
forecast_gen_other     = forecast_gen_total - forecast_gen_wind_pv
forecast_residual_load = forecast_load - forecast_gen_wind_pv
```

### Historical waterfall for training rows

For historical rows, construct `forecast_*` with this availability order:

```text
own gen/load model forecast -> SMARD forecast -> actual
```

This is a pragmatic pre-coverage bridge, not a statement that SMARD D+1 values are valid for live production at 08:00 UTC. It lets older training rows remain usable before own forecast artifact coverage begins.

Actual fallback definitions must be explicit:

```text
forecast_gen_wind_pv   = actual wind_on + actual wind_off + actual solar
forecast_gen_total     = actual total generation, or sum of actual generation components
forecast_gen_other     = forecast_gen_total - forecast_gen_wind_pv
forecast_residual_load = actual load - forecast_gen_wind_pv
```

### Deployment rule for live D+1 rows

For the 24 D+1 price forecast rows, own gen/load forecasts must be complete and are the only valid inputs. SMARD D+1 values and actuals are not available at 08:00 UTC and must never be used, even as a fallback.

`build_forecast_columns` must accept a `strict_index` parameter. When set to the D+1 delivery index:

- Read the five own forecast artifacts listed above.
- Check that all five source artifacts have non-NaN `y_pred` for **every hour in `strict_index`** (assert against `strict_index` hour-for-hour rather than a literal count — see DST note below).
- Raise if any source artifact is missing or incomplete.
- Do not read, forward-fill, or fall back to SMARD `prognostizierte_*` values for the strict rows.
- Derive `forecast_gen_wind_pv`, `forecast_gen_total`, `forecast_gen_other`, and `forecast_residual_load` from the strict primary values.

The check must happen on the five source artifacts individually. Checking only derived columns can hide a stale or incomplete primary input.

**DST — the real issue is a dropped 02:00 label, not a row count.** Correcting an earlier draft of this plan: this repo does **not** use 23/25-hour DST days. Both `merged.parquet` (tz-naive Europe/Berlin wall-clock labels) and the forecast artifacts (tz-aware UTC) carry a full **24** labels per day, verified on all recent DST dates. `_extend_to_forecast_date` builds `d1_index = date_range(00:00, 23:00, freq="h")` = 24 labels always. So assert coverage hour-for-hour against `strict_index`, not a hardcoded 24 — but do not expect 23 or 25.

There is, however, a genuine one-day-per-year bug to resolve, and its fix is **decided: Option A — DST-normalize the artifact grid the same way the training grid is normalized.** On the **spring-forward** delivery day (e.g. 2025-03-30), `_align_tz` converts the UTC artifact to Europe/Berlin and strips tz; the 02:00 wall-clock hour does not exist that day (clocks jump 01:59→03:00), so the aligned artifact has no 02:00 label. Reindexing onto the 24-label `d1_index` (which *does* contain 02:00) yields exactly **1 NaN at 02:00** — verified empirically. As specified, unmodified strict mode would raise on that single day each year and the daily forecast would fail.

The resolution is to reuse the repo's existing, tested DST normalization rather than special-case the strict gate. `merged.parquet` — the grid every price model was **trained on** — is already run through `normalize_dst()` (`data/merge.py:325`, called at `merge.py:729`, itself a port of upstream EP's `normalize_dst` in `ts_transforms.py`), which enforces exactly 24 local hours per day:

- **Spring-forward (23h day):** insert the missing 02:00 as the mean of 01:00 and 03:00.
- **Fall-back (25h day):** average the duplicate hour into one.

The builder must apply this **same** normalization to the five forecast artifacts (inside `build_forecast_columns`, on each primary series after `_align_tz`, before deriving `forecast_gen_total`/`forecast_gen_other`/`forecast_residual_load`). After it, the artifact grid matches `merged.parquet` exactly (24 rows, 02:00 present), strict mode sees no NaN, and the inference grid is identical to the training grid — no train/inference skew.

Why Option A over the alternatives: dropping 02:00 (a 23-row day) mismatches the 24-row training grid and breaks the "24 rows/day" assumption baked into `publish.py:468`, `validation.py`, the API, and the dashboard; merely tolerating the NaN in the strict gate still requires synthesizing a 02:00 value (so you interpolate anyway) while adding a special case that can mask real gaps. Option A also gives comment-4 source coherence for free: the 02:00 interpolation is applied uniformly across all five primaries **before** the differences are taken, so the derived quantities stay internally consistent. (The autumn fall-back day is already clean once normalized: the repeated hour is averaged → 0 NaN, verified.)

**Precondition — the five artifacts must carry live D+1 rows.** Strict mode only works if the daily gen/load inference has already appended that morning's D+1 `y_pred` to all five artifacts before price inference runs. This holds today: `gen_load_inference.py` writes `gen_load_diff` in wave 3 via `update_historical_forecasts`, and CLAUDE.md confirms `DE_NATIONAL` aggregates are written. But the plan depends on it, so make it an explicit precondition: if the workflow ordering ever regresses so that price inference runs before wave-3 completes, strict mode fails every morning and the site goes dark. Guard the daily workflow order and note it in the CI comment fixed in Step 6.

### `pct_forecast_*` features

Replace `pct_prog_*` with `pct_forecast_*`. These features must divide by `forecast_gen_total`, not by raw `prognostizierte_erzeugung_gesamt`, so numerator and denominator come from the same availability contract.

**The rename is not config-only.** `pct_prog_*` is not a stored column — it is *computed* in `features/market.py` (`add_prognosticated_pct`, ~lines 124-133, currently `df[prog_col] / prognostizierte_erzeugung_gesamt`) and plumbed through `features/engine.py` via the `_derived_pct_prog_` prefix (~lines 80, 87). Renaming the config feature lists without editing these two files leaves `pct_forecast_*` with **no producer** — training/inference will `KeyError` on the missing engineered column, or (if the old `pct_prog_*` computation is left in place) will silently keep dividing by the stale SMARD total, i.e. the exact bug this section claims to fix. The producer must be rewritten to divide the `forecast_*` numerators by `forecast_gen_total`. See Step 2.

---

## Raw vs Feature Columns

`merged.parquet` should retain raw SMARD `prognostizierte_*` columns for auditability and for historical waterfall fallback. They are not production price features.

The builder may create temporary/private intermediate columns, but it should not expose `_derived_forecast_*` names as price feature names. Either remove those names or keep them private inside the builder implementation.

---

## Implementation Steps

### Step 1 — Add `build_forecast_columns`

New file: `energy_forecasting/features/forecast_inputs.py`

```python
def build_forecast_columns(
    df: pd.DataFrame,
    *,
    strict_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
```

Behaviour:

- Load the five own forecast artifacts from `data/processed/historical_forecasts/*.parquet`.
- After `_align_tz`, DST-normalize each primary artifact series with the repo's existing `normalize_dst` logic (`data/merge.py:325`) so its grid matches `merged.parquet` (24 local hours/day; spring-forward 02:00 interpolated, fall-back duplicate averaged). Do this **before** deriving `forecast_gen_total`/`forecast_gen_other`/`forecast_residual_load`. See the DST note in the deployment rule (Option A).
- For normal historical rows, apply the waterfall: own forecast -> SMARD forecast -> actual.
- For `strict_index` rows, require complete own forecast artifact coverage and fail closed if incomplete.
- Derive all computed `forecast_*` columns from the primary forecast columns.
- Return the input DataFrame with the eight `forecast_*` columns appended.
- Do not mutate raw SMARD columns.

**Enforce per-row source coherence for the derived quantities.** The strict-index check guarantees all five artifacts for D+1, but historical rows can still mix layers: if one artifact has a gap, a row could combine own-load with SMARD-total, or own-wind/solar with actual-fallback others. Because `forecast_gen_total`, `forecast_gen_other`, and `forecast_residual_load` are *differences* of primaries, mixing sources within a single row injects artificial jumps that a model can latch onto as spurious signal. Adopt one of:
- **All-or-none per row (preferred):** for a given row, use the own-forecast layer only if all five primaries are present; otherwise fall the whole row back to the next coherent layer (SMARD, then actual). This keeps a row's primaries internally consistent.
- **Provenance-tagged:** emit per-column `forecast_source_*` audit data (own/smard/actual) so mixed rows are visible and testable.

Add a `test_partial_coverage_row` covering the case where one of the five artifacts has a hole.

This replaces `_overlay_ema_forecasts`. The old function should be removed rather than kept as a compatibility alias, because the old name preserves the wrong mental model.

### Step 2 — Update column and feature config

`energy_forecasting/config/columns.py`:

- Add `forecast_*` and `pct_forecast_*` short names.
- Remove old `prog_*` and `pct_prog_*` aliases from price feature use.

`energy_forecasting/config/features.py`:

- Replace every `prog_*` price feature with its `forecast_*` equivalent, including the operands inside interaction features (e.g. `prog_residual__x__day_index` → `forecast_residual_load__x__day_index`) and `_daily_*` variants (e.g. `prog_gen_total_daily_max`).
- Replace every `pct_prog_*` price feature with its `pct_forecast_*` equivalent.
- Add a comment that price feature lists must use source-neutral forecast names only.

`energy_forecasting/features/market.py` and `energy_forecasting/features/engine.py` (**required — this is where `pct_prog_*` is actually produced**):

- Rewrite `add_prognosticated_pct` in `market.py` to emit `_derived_pct_forecast_*` columns dividing the `forecast_*` numerators by `forecast_gen_total` (not `prognostizierte_erzeugung_gesamt`).
- Update the `_derived_pct_prog_` plumbing/prefix in `engine.py` (~lines 80, 87) to the new `_derived_pct_forecast_` names.
- Without this, the renamed `pct_forecast_*` features have no producer (see the `pct_forecast_*` contract note above).

`energy_forecasting/deploy/shap_attribution.py` (**required — silent breakage otherwise**):

- The driver-attribution map (~lines 38-39) keys categories on feature-name fragments including `prog_gen_solar`, `pct_prog_solar`, and `prog_gen_other`. Update these to `forecast_gen_solar`, `pct_forecast_solar`, `forecast_gen_other`. If left unchanged, the Stage 10 price-driver narrative silently loses its solar/residual matches and misclassifies drivers with no error raised.

### Step 3 — Update price dataset preparation

`energy_forecasting/modeling/price.py`:

- Update `prepare_price_dataset` to call `build_forecast_columns(df)` before feature engineering.
- `run_price_pipeline` should continue calling `prepare_price_dataset`; the forecast-column contract belongs in dataset preparation, not scattered through training code.

### Step 4 — Update price inference

`energy_forecasting/deploy/price_inference.py`:

- After `_extend_to_forecast_date`, call `build_forecast_columns(extended_df, strict_index=d1_index)`.
- Fix timing comments to state clearly that SMARD D+1 `prognostizierte_*` values are not available for the 08:00 UTC run.

**Do not blindly "remove the forward-fill."** There is no SMARD-specific ffill path to remove — `_extend_to_forecast_date` does a *blanket* forward-fill of every feature column (`price_inference.py:100-101`). Commodity features (carbon/ttf/brent, genuinely daily data whose last close is the correct D+1 value) and temporal placeholders legitimately depend on that ffill. Deleting it wholesale sends those columns to NaN and trips the NaN gate at `price_inference.py:268`, killing inference every day.

The blanket ffill is also *harmless* once price features reference `forecast_*` instead of `prog_*`: a stale forward-filled `prog_*` column is simply never consumed. So the safest change is:

- **Keep** the blanket ffill in `_extend_to_forecast_date`; leave its remaining job as extending the index, forward-filling legitimately-available features (commodities, temporal placeholders), and setting `target_price = NaN`.
- Let `build_forecast_columns(..., strict_index=d1_index)` **overwrite** the `forecast_*` columns for D+1 with own-model values (and raise if incomplete). The builder, not the ffill, owns the forecast columns.
- Optionally drop `prognostizierte_*` from the ffill set for clarity, but this is cosmetic — correctness comes from the builder overwriting `forecast_*` and from no price feature referencing `prog_*`.

### Step 5 — Delete or privatise confusing intermediate names

- Remove `_overlay_ema_forecasts` from `energy_forecasting/modeling/price.py`.
- Do not expose `_derived_forecast_*` names as price feature names. Remove them if possible, or keep them private inside `build_forecast_columns`.
- Remove helper functions made dead by the new builder — **but keep `_align_tz`** (`gen_load_forecasts.py:46-71`). The builder still reindexes tz-aware UTC forecast artifacts onto the tz-naive Europe/Berlin merged index; without `_align_tz` that reindex silently returns all-NaN. It moves with the builder, it does not die.

### Step 6 — Fix documentation and workflow comments

Update stale documentation that repeats the old SMARD availability assumption:

- `docs/stage6_inference_api.md`: replace the Stage 6 price inference section so it documents strict own forecast artifact coverage for D+1 and no SMARD D+1 fallback.
- `.github/workflows/daily_forecast.yml`: correct any cron comment that says the run happens after SMARD D+1 forecasts publish.
- Any status/master-plan text that says SMARD D+1 values are available before the 08:00 UTC public run.

Named active code comments that must not survive implementation (verified present):
- `energy_forecasting/config/columns.py:32` — "Forecasts (published for today — no lag required)" above the `prog_*` block. This is the false-availability claim in comment form; remove/correct it when the `prog_*` aliases go.
- `energy_forecasting/config/columns.py` — the older comment stating EMA forecasts overlay onto `prog_*` columns.
- `energy_forecasting/deploy/gen_load_inference.py` — `update_historical_forecasts` docstring: "These files are used by the price model's EMA overlay (`_overlay_ema_forecasts`) to supply the `prog_*` features for D+1." Update to reference `build_forecast_columns` and `forecast_*`.
- `energy_forecasting/deploy/price_inference.py:7, 229-237` — module docstring and inline comments describing applying the EMA overlay after D+1 extension.

### Step 7 — Fix and extend the existing leakage guard (do not add a parallel one)

The guard is not missing (see §4). The fix is to correct `AVAILABILITY_RULES` so the single source of truth stops blessing unavailable columns, then add a thin convenience check on top.

`energy_forecasting/config/availability.py`:

- **Remove or repurpose** the `prog_*` and `pct_prog_*` rules (currently lines 29, 40) that declare `max_offset_days=0`. Raw SMARD D+1 forecasts are not available to the 08:00 UTC price run, so they must not validate as `offset=0` price inputs.
- **Add** `forecast_*` and `pct_forecast_*` rules at `offset=0` — these are legitimately available at inference because own gen/load models produce them by 08:00 UTC. Give each a `reason` that states this explicitly.
- After this change, any `prog_*` / `pct_prog_*` name in a price feature list fails `validate_features()` for lack of a matching safe rule, from the mechanism that already runs in `engineer_features`.

`energy_forecasting/features/validation.py` (thin convenience layer, delegating to the same rules):

```python
PRICE_FEATURE_BANNED_PREFIXES = ("prog_", "pct_prog_", "prognostiziert")
```

`validate_price_feature_list(features: list[str])` raises if any banned source-derived feature (including as an interaction operand) appears. Call it before price dataset preparation and in tests. This is defence-in-depth for a fast, price-specific error message — it must not become the *only* guard, and it must not contradict `AVAILABILITY_RULES`.

**Guard the persisted feature schema too — config-only checks miss the runtime path.** Inference does not re-validate: `_engineer_features_for_version` runs `engineer_features(..., PRICE_FEATURES_MAX, validate=False)` and takes its column list from the persisted dataset schema or `price_feature_cols.json` (`price_inference.py:146-169, 179-183`), not from the config feature lists. So a stale `price_feature_cols.json` or dataset parquet still carrying `prog_*` columns would drive inference even after the config is clean. Add a release/runtime check that the *persisted* trained schema and exported `price_feature_cols.json` contain no `prog_` / `pct_prog_` / `prognostiziert` tokens, and make it an acceptance criterion. This is the artifact that actually feeds production.

---

## Retraining Plan

The existing price models were trained with leaky or availability-mismatched `prog_*` features. Full retraining is required after the feature contract changes.

1. Implement the forecast builder, feature renames, strict inference checks, leakage guard, and documentation fixes.
2. Regenerate merged data and price feature datasets.
3. Run a parity audit:
   - 2022+ rows with own forecast artifacts should use `forecast_*` values from artifact `y_pred`.
   - Pre-artifact rows may use SMARD forecast fallback.
   - Rows before SMARD forecast availability may use explicit actual fallback.
   - **Report provenance mix.** Count rows by source layer (own forecast / SMARD fallback / actual fallback) per year and per feature. Actuals in a model-ready `forecast_*` feature are a pragmatic pre-coverage bridge but are conceptually leaky, so make them visible: either exclude actual-fallback rows from feature-selection sensitivity checks or report their measured impact.
   - **Plot the construction boundary.** `forecast_gen_total` is SMARD-published before ~2022 and `forecast_load + gen_load_diff`-derived after; the two constructions have different bias/noise. Plot the series across the ~2022 boundary so the structural break is a known quantity, not a surprise the model discovers as spurious signal. Holdout is recent (post-boundary), so this is acceptable — but confirm it.
4. Re-run RFECV and SHAP feature selection.
5. Retrain base price models on the new feature sets.
6. Re-blend ensemble weights.
7. Re-calibrate conformal intervals.
8. Backtest with special focus on high-solar days, negative-price hours, and large residual-load ramps.
   - **Not every `forecast_*` is a strict upgrade.** `forecast_residual_load` is a large win versus the old flat/stale value. But `forecast_gen_other = (forecast_load + gen_load_diff) − (wind_on + wind_off + solar)` and `forecast_gen_total` replace a *single directly-published* SMARD quantity with a difference of ~4 noisy model outputs, so their noise floor is materially higher than the SMARD original. Check per-feature that these are helping, not injecting noise that outweighs the residual fix; be prepared to drop `forecast_gen_other`/`forecast_gen_total` if feature selection or an ablation says they hurt.
9. Release new model artifacts only after the backtest is understood.
   - **Do not treat the 11.148 EUR/MWh holdout number as a target to beat.** That benchmark was measured on holdout rows where the leaky SMARD `prog_*` values were present — a figure never achievable in live production. After the fix, holdout uses own-model `forecast_*` (noisier than TSO operator forecasts), so an *honest* holdout MAE will likely be numerically **worse** than 11.148, and that is the fix working, not failing. The meaningful comparison is production forecast error before vs after the fix. If a holdout baseline is wanted, recompute the *old* pipeline's holdout MAE using only inference-available inputs (own forecasts, not SMARD D+1) to get an honest number to compare against.

---

## Feature Name Cleanup

| What | Action |
|---|---|
| `prog_*` entries in price feature lists (incl. interaction operands, `_daily_*`) | Replace with `forecast_*` equivalents |
| `pct_prog_*` entries in price feature lists | Replace with `pct_forecast_*` equivalents |
| `add_prognosticated_pct` in `features/market.py` + `_derived_pct_prog_` plumbing in `features/engine.py` | Rewrite to produce `_derived_pct_forecast_*` dividing by `forecast_gen_total` |
| `prog_*` / `pct_prog_*` availability rules in `config/availability.py` (lines 29, 40) | Remove/repurpose; add `forecast_*` / `pct_forecast_*` rules at `offset=0` |
| Driver-attribution fragments in `deploy/shap_attribution.py` (`prog_gen_solar`, `pct_prog_solar`, `prog_gen_other`) | Rename to `forecast_*` equivalents |
| Raw `prognostizierte_*` columns | Keep in merged data for audit/fallback; ban from production price feature lists |
| `price_feature_cols.json` | Regenerate during retraining |
| Old price model artifacts | Archive or replace only after new backtests pass |
| `_overlay_ema_forecasts` | Delete |
| `_derived_forecast_*` exposed names | Remove or keep private behind the builder |
| `_align_tz` (`gen_load_forecasts.py`) | Keep — still needed by the builder for tz-aware→tz-naive reindex; follow it with `normalize_dst` so the artifact grid matches `merged.parquet` (Option A) |
| `normalize_dst` (`data/merge.py:325`) | Reuse in the builder for artifact DST normalization; do not fork a second implementation |
| Incorrect SMARD timing comments | Fix in code, workflow comments, and docs |

Gen/load feature config does not need the same rename if it only uses lagged actual features such as `gen_wind_on_h24`. Audit any `prog_*` use in gen/load contexts separately and judge it by actual availability.

---

## Regression Tests

1. `test_no_prog_in_price_features`: assert no `PRICE_FEATURES_*` entry contains a `prog_` or `pct_prog_` token **anywhere** (use a token/substring check, not just `startswith`, so an interaction operand like `x__x__prog_gen_other` is also caught) and no entry references raw `prognostizierte*` names.
2. `test_forecast_column_derivations`: with synthetic load=50, wind_on=10, wind_off=5, solar=8, gen_load_diff=3, assert `forecast_gen_wind_pv=23`, `forecast_gen_total=53`, `forecast_gen_other=30`, and `forecast_residual_load=27`.
3. `test_pct_forecast_divides_by_forecast_total`: assert `pct_forecast_*` numerators are divided by `forecast_gen_total`, not by raw `prognostizierte_erzeugung_gesamt` (guards the `market.py`/`engine.py` producer change).
4. `test_waterfall_prefers_own_forecast`: if own forecast and SMARD values both exist for a historical row, assert the `forecast_*` value comes from own forecast artifact `y_pred`.
5. `test_waterfall_falls_back_to_smard`: for a row without own forecast coverage but with SMARD forecast coverage, assert the `forecast_*` value comes from SMARD.
6. `test_waterfall_falls_back_to_actual`: for a row without own forecast or SMARD forecast coverage, assert the explicit actual fallback formulas are used.
7. `test_strict_mode_raises_on_missing_source`: strict builder fails if any of the five own forecast artifacts lacks complete coverage over `strict_index`.
8. `test_strict_mode_spring_forward`: build a strict `d1_index` for the real spring-forward date (e.g. 2025-03-30, 24 tz-naive labels including 02:00). Assert Option A behaviour: the builder returns 24 complete rows with 02:00 present and equal to the mean of the 01:00 and 03:00 primaries (interpolated, not NaN), and strict mode does not raise. Also cover the fall-back date (e.g. 2025-10-26) to confirm the duplicate hour is averaged to a single clean row. Do not use synthetic 23/25-hour indices.
9. `test_strict_mode_no_smard_fallback`: strict builder does not read or forward-fill SMARD `prognostizierte_*` values.
10. `test_availability_rule_rejects_prog`: after the `availability.py` change, `validate_features(["prog_residual"])` returns an error (the single-source-of-truth guard, not just the convenience wrapper).
11. `test_leakage_guard`: `validate_price_feature_list` raises on `prog_residual`, `pct_prog_solar`, or raw `prognostizierte_*` entries, including as interaction operands.

---

## Acceptance Criteria

1. No `PRICE_FEATURES_*` entry contains a `prog_` or `pct_prog_` token (including interaction operands).
2. `config/availability.py` no longer blesses `prog_*`/`pct_prog_*` as `offset=0`, and `validate_features()` rejects a `prog_*` price feature from the single source of truth.
3. `pct_forecast_*` is produced by `market.py`/`engine.py` dividing by `forecast_gen_total`, and `pct_prog_*` has no remaining producer.
4. Strict builder fails if any of the five own forecast artifacts lacks complete coverage over `strict_index`, validating hour-for-hour against `strict_index` (not a hardcoded 24). On the spring-forward delivery day it returns 24 complete rows with an interpolated 02:00 (Option A: artifacts DST-normalized via `normalize_dst`), and does not raise.
5. Exported `price_feature_cols.json` and every persisted trained dataset schema contain no `prog_` / `pct_prog_` / `prognostiziert` token (checked at release, not just in config).
6. The builder never mixes source layers within a single row's derived quantities (all-or-none per row, or provenance-tagged and audited).
7. Strict mode does not read or forward-fill SMARD `prognostizierte_*` values; the blanket ffill for legitimately-available features (commodities, temporal) still runs so those columns are not NaN.
8. Latest daily feature audit shows required D+1 `forecast_*` columns complete for every delivery hour.
9. `deploy/shap_attribution.py` driver map references `forecast_*` names, and Stage 10 price-driver attribution still resolves solar/residual drivers.
10. Backtest report includes high-solar and negative-price slices, and states the honest holdout MAE with the explicit note that a regression versus the old 11.148 figure is expected (that number was leakage-inflated).
11. `docs/stage6_inference_api.md` no longer documents SMARD D+1 values as available to the 08:00 UTC forecast run.

---

## Expected Benefit

- Removes the price model's dependency on unavailable SMARD D+1 forecast values.
- Makes feature names encode availability semantics: `forecast_*` means valid under the forecast waterfall and strict live-inference rules.
- Makes historical fallback explicit without confusing it with live D+1 production behaviour.
- Adds a guardrail so future feature-list changes cannot silently reintroduce raw SMARD forecast columns.
- Gives future agents and maintainers a clearer contract to follow.

---

## Implementation Notes

### 2026-07-13 — checkpoint `df9206c` (`Fix price forecast feature inputs`)

Implemented the core forecast-column contract in code and tests:

- Added `energy_forecasting/features/forecast_inputs.py` with `build_forecast_columns()`. It constructs the eight source-neutral `forecast_*` columns from a coherent waterfall: own gen/load forecast artifacts first, then SMARD forecasts, then actuals.
- Live D+1 inference now calls `build_forecast_columns(..., strict_index=d1_index)` after `_extend_to_forecast_date`. Strict mode validates all five own forecast artifacts hour-for-hour and fails closed if any primary artifact is missing.
- Historical rows use an all-or-none own-artifact layer. If any one of the five primary own artifacts is missing for a row, that row falls back as a unit to the next coherent layer rather than mixing own load with SMARD/actual generation. The builder also records `forecast_source_counts` in `DataFrame.attrs` for audit.
- Artifact DST handling follows Option A from this plan for tz-aware artifact inputs: `normalize_dst()` is reused so the artifact grid matches the 24-row local delivery grid used by `merged.parquet`. A local-naive fallback normalizer handles duplicate/missing local labels defensively.
- Added an optional `forecast_root` keyword to `build_forecast_columns()` for tests and controlled backfills. This is a small, non-production signature extension from the original plan; production still uses `HISTORICAL_FORECASTS_DIR`.
- Replaced price feature-list usage of `prog_*`/`pct_prog_*` with `forecast_*`/`pct_forecast_*`, including interactions and daily aggregates. Raw `prog_*` aliases remain in `SHORT_NAMES` only for audit/backward-compatible parsing and no longer validate as safe production features.
- Rewrote `compute_generation_pct()` and `features.engine` plumbing so `_derived_pct_forecast_*` divides by `forecast_gen_total`, not raw SMARD `prognostizierte_erzeugung_gesamt`.
- Removed `_overlay_ema_forecasts` from `modeling/price.py`; training dataset prep now calls `build_forecast_columns(df)` before feature engineering.
- Removed the old EMA-overlay mutation step from `deploy/retrain.py`. Retrain now assumes price datasets have been rebuilt through `prepare_price_dataset()` so they already contain the new forecast-column contract.
- Added persisted-schema protection in `deploy/price_inference.py`: trained dataset schemas and `price_feature_cols.json` are rejected at runtime if they contain `prog_`, `pct_prog_`, or raw `prognostiziert` tokens.
- Updated `deploy/shap_attribution.py` driver fragments to recognize `forecast_gen_solar`, `pct_forecast_solar`, and `forecast_gen_other`.
- Updated active code comments/docstrings that described the old overlay or claimed SMARD D+1 forecast availability.

Validation run for this checkpoint:

```text
conda run -n energy-forecasting pytest tests/test_forecast_inputs.py tests/test_5c_derivations.py tests/test_engine.py tests/test_parser.py tests/test_validation.py tests/test_availability.py tests/test_deploy_price_inference.py tests/test_deploy_shap_attribution.py
147 passed in 2.23s
```

Important discoveries and decisions:

- Missing artifact files initially used `pd.NA`, which failed float casting before historical fallback could run. This was fixed to use numeric NaN so non-strict historical rows can fall back while strict D+1 rows still fail closed.
- Keeping the raw `prog_*` aliases in `SHORT_NAMES` is intentional. Removing them entirely would make old references fail at parse time, but keeping them without availability rules gives a clearer validation error and preserves audit/backfill tooling flexibility.
- Existing deployed/trained model artifacts are expected to fail the new persisted-schema guard until price datasets and production models are regenerated. That is deliberate: stale `prog_*` schemas are unsafe under the new contract.
- This checkpoint does not perform the required full price retrain, RFECV/SHAP reselection, ensemble reblend, conformal recalibration, or production artifact release. Those remain required before daily production inference can use the new feature contract end to end.
