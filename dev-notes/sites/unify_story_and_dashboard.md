# Unify the story site + dashboard, and fix outstanding bugs (staged)

## Context

Two public surfaces sit on the same forecasting models:

- **Forecast story deck** — `deploy/story/forecast/index.html`, a one-slide-at-a-time Plotly deck
  with the more developed visual system (design tokens, light/dark, narrative boxes). *This* becomes
  the main entry point.
- **Dashboard** — `deploy/index.html` + `script.js` + `styles.css`, a three-tab (Prices /
  Generation & Load / Monitoring) app. Light-only, Bootstrap greys, no explanatory text, several
  stale/broken monitoring panels.
- (History scrollytelling page `deploy/story/index.html` is **not ready** — stays unlinked.)

Goal: story deck as landing with a prominent dashboard link, dashboard restyled to the story's
standard, a shared manual light/dark toggle, and a batch of content + data fixes on both surfaces.

Work is split into **stages with verification gates** — each stage is independently checkable and we
don't advance past a broken gate. **All copywriting is delegated to a focused sub-agent** (Stage 6),
because drafting inside a large coding task produces weak text.

## Decisions locked (from Q&A)

- **Entry structure:** story deck = landing; root redirects to it; dashboard is a prominent
  secondary link. History links removed until ready.
- **CV-MAE monitoring chart:** remove entirely (linear-beats-trees is a full-history-vs-recent-window
  artifact and misleads).
- **Luxembourg:** **restore EMA fidelity** — add `DE_CREOS` to `wind_onshore` + `solar` and retrain
  those two targets. EMA models LU wind/solar per-region (`eu_locations.py:1784-1785`); our merge
  dropped it. Weather + target columns already exist on disk. `gen_load_diff` is zone-level and
  needs no LU change. Impact <0.1% but it restores upstream fidelity and fixes the renewable/"other"
  misclassification (LU's ~25 MW currently lands in "other generation").

## Root-cause findings that shape the work

- **Load → zero (both surfaces).** Primary: `aggregate_national()` sums per-TSO frames with
  `sum(df.reindex(all_idx, fill_value=0) ...)` (`deploy/gen_load_inference.py:474`); short per-TSO
  horizons (trimmed in `_infer_one`, `:295-323`, when a weather/exog forecast tails off to NaN) get
  0-padded, so the national total collapses at the edge. Secondary: recent actuals sum with
  `min_count=1` (`publish.py:557`, `:662`), so late-reporting TSOs contribute 0. Load worst hit
  (5 TSOs). **Fix:** NaN-preserving aggregation, never 0; omit NaN points on publish.
- **"Story site" = the forecast deck.** Light/dark already works via `@media` (`story.css:25-46`);
  the ask is a *manual* toggle, not a new theme.
- **CV-MAE inversion:** `cv_mae` copied verbatim from full-history rolling folds (`publish.py:444`).
- **Monitoring noise:** pre-July OOF/backfill rows in `gen_load_errors_summary.json`
  (`write_gen_load_errors`, `publish.py:692-711`, `.tail(30)`, no date floor).

---

## Stage 1 — Backend data/pipeline fixes  `energy_forecasting/deploy/`

1. **Load-to-zero — `aggregate_national()`** (`gen_load_inference.py:465-474`): replace
   `sum(reindex(..., fill_value=0))` with NaN-preserving aggregation — reindex each frame to the
   union index without a fill, concat column-wise, sum with `min_count = len(frames)` so any hour
   missing a TSO is NaN. Emit NaNs; never zero-fill. Check whether `_infer_one` horizon trimming
   (`:295-323`) can be avoided by extending exog/weather coverage; if not, the NaN aggregation makes
   the series end cleanly rather than dive.
2. **Partial-TSO actuals** (`publish.py:557`, `:662` in `write_gen_load_actuals`): `min_count =
   len(series_list)`; drop NaN hours in `_series_to_days` (`:527-543`) instead of appending
   under-summed values.
3. **Monitoring noise — `write_gen_load_errors`** (`publish.py:692-711`): drop `date < "2026-07-01"`
   before the tail, and window to **14 days**.

**Gate 1:** run gen/load inference; confirm `load_national.json` + per-TSO/actuals JSON have no
trailing zeros (national load ~44-60 GW to horizon end); `gen_load_errors_summary.json` starts
≥2026-07-01 with ≤14 points. Deploy test suite green (512 baseline).

---

## Stage 2 — Luxembourg wind/solar retrain (EMA fidelity)

*Heavy compute — launch detached per `~/.claude/CLAUDE.md` §Long-running processes. Can run in the
background while Stages 3-5 proceed; its gate is independent.*

4. Add `DE_CREOS` to `wind_onshore.regions` and `solar.regions` in `config/modeling.py:92-102`; add
   `"Creos"` to the matching `_TARGET_TSO_FILES` entries (`publish.py:70-73`); add `"creos"` to the
   wind_onshore/solar `GEN_LOAD_CARDS.tsos` (`script.js:12-37`). Add an inline comment at
   `GEN_LOAD_TARGETS` recording the EMA-fidelity rationale; correct the stale
   `dev-notes/stage2_data_collection.md:1631` note.
5. Confirm Creos weather (`data/raw/weather/{cities,onshore,solar}/Creos/`) + target columns
   (`wind_onshore_lu`, `solar_lu` in `Creos.parquet`) are present, then run the gen/load training for
   `wind_onshore` and `solar` (respecting the wind/solar → load → gen_load_diff order) to produce the
   two new per-region models + their historical_forecasts.

**Gate 2:** LU models exist; national wind/solar aggregation now folds in Creos; the renewable/"other"
split moves ~25 MW from "other" into wind/PV; no regression on the four German TSO models.

---

## Stage 3 — Shared theming + entry structure

6. **Manual light/dark toggle (shared).** Keep `@media (prefers-color-scheme: dark)` as default but
   add `:root[data-theme="dark"]` / `[data-theme="light"]` overrides that win; a small shared inline
   theme script (read `localStorage`, fall back to system pref) + a toggle button in the story nav
   and dashboard header. Re-plot/relayout charts on toggle so Plotly colors (read via `cssVar()`)
   follow.
7. **Story = landing.** Rename dashboard `deploy/index.html` → `deploy/dashboard.html` (stays in the
   same dir, so its `data/…`, `styles.css`, `script.js`, `translations.json` fetches keep working);
   new `deploy/index.html` redirects to `story/forecast/index.html`. Update internal links:
   `monitoring.html` → `dashboard.html#monitoring`; story footer `../../index.html` →
   `../../dashboard.html` (`forecast/index.html:159`); dashboard header story link →
   `story/forecast/index.html`. Grep workflows + `deploy/` for hardcoded `deploy/index.html` first.

**Gate 3:** root redirects to the deck; `dashboard.html` loads with working data/assets; the story
deck toggle flips the theme and re-colors charts; no dead history links (grep `story/index.html`).

> **Deviation (recorded during Stage 3):** the shared toggle mechanism (`deploy/theme.js`, the
> pre-paint `<head>` init, and `:root[data-theme=...]` token overrides) is built now and wired to the
> **story deck**. The **dashboard's** toggle button + dark tokens are deferred to **Stage 5**, because
> the dashboard currently has *no* dark palette (hard-coded hex, no tokens) — adding a toggle before
> the Stage 5 restyle would flip it into a broken half-dark state. `theme.js` already targets every
> `[data-theme-toggle]` and dispatches `themechange`, so Stage 5 only adds the button + a rebuild
> listener on the dashboard side. Checklist A2 already attributes the dashboard toggle to Stage 5.

---

## Stage 4 — Story deck structure/code (non-text)  `deploy/story/forecast/`

8. **Fit charts to viewport.** Replace fixed inline `height:440/460px` on chart divs with
   viewport-relative heights (`clamp()` / `min(vh, px)`) so chapter head + panel + narrative +
   `.deck-controls` fit one screen. Resize already wired (`index.html:192-209`).
9. **Split the two-chart slide.** `#forecast-price` (`index.html:100-122`) → two slides (price
   forecast; then "what's driving it" SHAP); add to deck/dots/nav.
10. **Forecast vs actuals overlap + load dip.** In `buildForecastGenLoadCharts`
    (`forecast-story.js:295-367`): after Stage 1, the load tail stops diving. Share a continuous
    x-range so actual (gray) meets forecast (blue) at the seam; overlay `gen_load_hindcast.json`
    across the recent actual window so the two genuinely overlap for validation (dashboard-card
    pattern).
11. **Track-record → last 7 days.** `buildTrackRecordChart` (`forecast-story.js:558-603`) currently
    concatenates every issued forecast; slice to the last 7 delivery days.
12. **Remove history links, add dashboard link.** Drop nav "The history →" (`index.html:22`) + closing
    CTA (`index.html:139`); add a prominent, larger "See the numbers → live dashboard" button.

**Gate 4:** every slide fits one screen; price + SHAP on separate slides; actuals meet forecast with
no zero dip; track-record shows 7 days; no history links; dashboard link prominent. (Copy still
placeholder — filled in Stage 6.)

---

## Stage 5 — Dashboard structure/code (non-text)  `deploy/`

13. **Restyle to the story system.** Port `story.css` `:root` tokens + dark block into the dashboard
    CSS; replace hard-coded hex with `var(...)`; remap JS color constants (`CATEGORY_COLORS`,
    `MODEL_PALETTE`, `TSO_COLORS`, `GL_COLORS`, `GEN_LOAD_CARDS[].color`, `script.js:12-68`) to the
    story palette.
14. **Remove CV-MAE panel** — `renderErrorTrend` + `#error-trend-chart` (`script.js:784-812`) + markup.
15. **Cut Deployment Feature Audit** — `renderFeatureAudit` + `#feature-audit-panel`
    (`script.js:975-1039`) + markup; leave a labelled placeholder for the future data-collection
    flags panel (**deferred**, not built now).
16. **Gen/load accuracy chart** (`renderGenLoadErrors`, `script.js:919-963`): rely on Stage 1's
    server-side filter; drop the client-side OOF-divider hack; label "last 14 days."
17. **Add gen_load_diff panel** on the Generation & Load tab from `gen_load_diff_national.json`
    (net balance, small ±); forecast-only by default.
18. **Reconcile title/window mismatches** (History H2 "30 Days" vs 7-day slice; Model-MAE "30 Days"
    vs full history).

**Gate 5:** dashboard matches story palette in both themes; CV-MAE + feature-audit gone;
gen_load_diff panel present; gen/load-accuracy readable (14 days); LU/Creos now on wind/solar cards
(post-Stage 2). (Copy/captions still placeholder — Stage 6.)

---

## Stage 6 — Copywriting pass (**delegated to a focused sub-agent**)

Give a sub-agent a *focused* brief — the finalized slide/panel structure from Stages 4-5, the
`facts_yearly.json` schema, the existing sentence functions, and a strict tone spec (professional,
factual, no overpromising/interpretation) — and have it draft:

- **Hero** (`forecast/index.html:27-39`): shorter H1 (kill the doubled "volatile"), a factual lede,
  a compact headline-stat treatment for visual interest.
- **Narrative sentence templates** (`forecast-story.js:210-285`, `:469`, `:492`, `:547`): rewrite the
  templated `*Sentence()` logic to read clearly — fix "wind up 2.5 points and coal down 2.3 points"
  (`:244-251`) to e.g. "wind's share rose 2.5 pp to X%, coal's fell to Y%"; strip interpretation.
- **Chart captions / explanatory text**: a `.sub`-style one-liner under every dashboard chart H2 and
  any story panel missing one, saying what the chart is.
- **Luxembourg note**: short wording on the dashboard that LU/Creos is forecast per-region and its
  generation is small.

Sub-agent returns copy + revised template logic; I integrate. Optionally run `/check-writing` on the
prose.

**Gate 6:** user reviews the copy; narrative boxes render real sentences (no undefined/NaN) against
fixture JSON; sentence-template unit tests updated/passing.

---

## Stage 7 — End-to-end verification + release

Regenerate deployed JSON with a real inference run (so monitoring + gen/load panels render against
fixed data, not stale copies). Walk both surfaces in light + dark. Run the full suite. Commit in
logical chunks. **Push before prod picks up the Stage 2 models** — per the deploy-skew outage note,
new models releasing while fix code sits unpushed caused a multi-day outage.

## Out of scope / deferred

- The full data-collection **flags** monitoring panel (placeholder only now).
- History scrollytelling page — unchanged, unlinked.
