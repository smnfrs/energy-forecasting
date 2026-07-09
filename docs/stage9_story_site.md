# Stage 9: Public Storytelling Site

A scrollytelling site that explains the German (DE-LU) electricity market to a lay audience, using the project's own historical data. Distinct from Stage 7's dashboard: Stage 7 is an operational tool for people who already want the forecast; Stage 9 is a narrative front door for people who don't yet know why the forecast matters. The two cross-link.

**Detailed plan:** this document
**Status:** steps 1-5 done (2026-07-07) — plan approved, `scripts/build_story_data.py` + `make story-data` computing all 6 datasets, chart prototypes validated in `deploy/story/chart_prototypes.html`, and the scrollytelling shell built in `deploy/story/index.html` with all 7 chapters wired (hook, prices, gas shock, neighbours, renewables, negative prices, forecast CTA) using Scrollama for nav-highlighting and a sticky-graphic/scrolling-copy two-column layout (single column on mobile). Step 7 (cross-link) also done early: `deploy/index.html` header links to the story, the story's CTA/footer link back. Steps 6 (narrative copy is a first draft, not yet user-reviewed for tone), 8 (full responsive/a11y QA), 9 (Playwright smoke test) remain.

---

## Context

Reference: https://dearricharddawkins.com/ — a single scrolling narrative broken into chapters, each chapter pinning a visual in a sticky pane while text scrolls past it (classic "scrollytelling" pattern, e.g. Pudding/NYT features), plus a persistent tab/nav bar so a reader can also jump straight to a chapter instead of scrolling.

Data check performed before writing this plan — everything below is confirmed present in `data/processed/merged.parquet` (2014-12-31 → 2026-07-01, hourly, no gaps in the checked columns):

| Story need | Column(s) | Notes |
|---|---|---|
| DE-LU day-ahead price history | `target_price` | range -500 to 936.28 EUR/MWh |
| Gas price shock | `ttf_eur_per_mwh` | annual mean: 2020 €9.6 → 2021 €47.1 → **2022 €132.1** → 2023 €41.5 → 2026 €42.9 |
| Carbon price | `carbon_eur_per_ton` | secondary overlay, optional |
| Neighbouring bidding zone prices | `marktpreis_{oesterreich,belgien,frankreich,niederlande,polen,schweiz,italien_(nord),tschechien,daenemark_1,daenemark_2,norwegen_2,schweden_4,slowenien,ungarn}` | DE-AT decoupled Oct 2018 — worth a callout |
| Generation mix / renewable share | `stromerzeugung_{wind_offshore,wind_onshore,photovoltaik,biomasse,wasserkraft,sonstige_erneuerbare,braunkohle,steinkohle,erdgas,kernenergie,pumpspeicher,sonstige_konventionelle}` | renewable share rose 2015 32.4% → 2025 57.1% |
| Negative price events | `target_price < 0` | hours/year: 2015: 110, 2020: 298, 2022: 69 (crisis suppressed them), 2023: 301, 2024: 457, 2025: 573 — clear rising trend post-2023 |

No new data collection needed for the first cut. Everything is a re-aggregation of data Stage 2-4 already produced.

### Relationship to existing dashboard

`deploy/` currently serves `index.html` (Prices / Gen & Load / Monitoring tabs) via the existing Stage 6 GitHub Pages workflow (`actions-gh-pages`, `publish_dir: ./deploy`). Adding the story site as `deploy/story/` rides that same workflow with **zero CI changes** — same pattern Stage 7 used. The story site is static history; it does not need daily refresh like the forecast dashboard does, so it does not need to hook into `publish.py` or the daily inference workflow.

---

## Decisions

**Location: `deploy/story/` subfolder**, not a new top-level directory or separate repo. Reuses the existing Pages publish step, keeps one deployment surface, and both sites can link to each other with relative paths (`../index.html` / `story/index.html`).

**Scroll mechanics: Scrollama.js** (CDN, ~5kb, MIT) rather than hand-rolled `IntersectionObserver` code. It's the standard tool for exactly this pattern (step-enter/step-exit callbacks tied to sticky panels) and keeps us at zero-build-step, matching the Plotly-via-CDN posture already used in `deploy/`.

**Charts: Plotly**, consistent with the existing dashboard, for the same reason Stage 7 chose it (PI/band fills, one library for the whole `deploy/` tree, no new dependency).

**Data refresh: manual/periodic, not daily CI.** A `make story-data` target regenerates the static JSON on demand. The story is about long-run patterns, not yesterday's forecast — wiring it into the daily workflow would be effort spent on freshness nobody needs. Revisit only if we later want e.g. "negative price hours this year so far" to stay current.

**Content ownership:** I'll draft narrative copy per chapter grounded in the numbers above, but tone/voice for public-facing text is the user's call to edit — flagged as its own review step below, not bundled into chart-building.

---

## Narrative arc

1. **Hook.** Germany's electricity market has gone from boring to genuinely wild — frame the "why should I care" in one line + a striking headline number (e.g. the €936 vs -€500 price range).
2. **A decade of prices.** Daily mean day-ahead price, 2015-2026. Establishes the baseline before the story complicates it.
3. **The 2022 gas shock.** TTF gas price overlaid on power price — the ~14x gas spike and its pass-through to power prices. Carbon price as a secondary, lighter line.
4. **Germany vs its neighbours.** DE-LU price next to AT/FR/NL/PL/CH etc. Coupling in normal times, divergence during the crisis; the DE-AT bidding-zone split (2018) as a concrete historical marker.
5. **The renewable rise.** Stacked generation mix (annual/monthly resample) + renewable-share line, 32% → 57%.
6. **Negative prices.** Rising frequency (bar per year) + one real example day (hourly price + generation) showing the duck-curve mechanism — high renewables, low demand, negative price.
7. **Why forecasting matters.** Volatility + renewables growth is exactly what makes day-ahead prices hard to predict — segue into what this project does, with a CTA link to the live dashboard.
8. **Footer.** Methodology, data sources (SMARD, ENTSO-E-style neighbour prices, TTF/carbon/Brent), credits, link back to the dashboard and repo.

---

## Chart list (first cut)

| # | Chart | Data | Chapter |
|---|---|---|---|
| A | Daily-mean price line, 2015-2026 | `target_price` resampled daily | 2 |
| B | Gas vs power price, two stacked panels sharing one time axis (not dual-axis — see design notes) | `ttf_eur_per_mwh`, `target_price` | 3 |
| C | Bidding-zone comparison, DE-LU vs selected neighbours, toggleable lines | `marktpreis_*` | 4 |
| D | Generation mix stacked area + renewable-share line | `stromerzeugung_*` | 5 |
| E1 | Negative-price hours per year, bar | `target_price < 0` | 6 |
| E2 | Single example day, hourly price + generation (duck curve) | `target_price`, `stromerzeugung_*` for one chosen date | 6 |
| F (stretch) | Forecast-vs-actual teaser, sampled from Stage 6 output | `deploy/data/price_forecast.json` / `forecast_history.json` | 7 |

Charts A-E are all buildable today from `data/processed/merged.parquet` alone.

### Design notes (per the dataviz skill)

- **No dual-axis charts anywhere** — gas-vs-power (B) and the duck-curve day (E2) each use two stacked panels sharing one time axis instead, since a single overlaid plot with two y-scales invents a correlation that isn't really there.
- **Generation mix (D) is folded to 7 categories** to stay under the categorical-color ceiling: Wind (onshore+offshore), Solar, Hydro & biomass, Nuclear, Natural gas, Coal (lignite+hard coal), Other (pumped storage + misc). Fixed stacking/legend order = renewables first (anchored at zero) so the rising green wedge is directly readable. Palette validated with `validate_palette.js` — passes in both light and dark mode (adjacent CVD ΔE ≥ 10.3), with a WARN on light-mode contrast for 3 slots that's satisfied by the legend + table-view fallback already built in.
- **Bidding-zone comparison (C) is small multiples, not one crowded multi-line chart** — DE-LU highlighted in blue in every facet, each neighbour in de-emphasis gray, one facet per country. Scales past the 7-8 series ceiling to as many neighbours as we want without adding hues.
- **Negative-price example day (E2) price panel uses diverging color** (blue positive / red negative) since sign relative to zero is exactly the polarity job.
- Every chart prototype ships a "view data as table" toggle (the accessibility non-negotiable), a shared crosshair tooltip (`hovermode: x unified`), and dark-mode CSS variables validated against the dark surface separately from light.

Prototype file: `deploy/story/chart_prototypes.html`. Data loading: `make story-data` writes `deploy/story/data/*.json` for reference/debugging, but both `chart_prototypes.html` and the real `index.html` load their charts from a `<script id="story-data" type="application/json">` data island that `build_story_data.py` inlines directly into each page — `fetch()` of a local JSON file is blocked under the `file://` protocol, so this lets either page be opened by double-clicking with no local server. Chart-building JS lives in the shared `deploy/story/charts.js`; shared tokens/CSS in `deploy/story/story.css`. Screenshotted in both color schemes via Playwright to check for layout/label issues.

---

## Implementation steps

1. **Plan review (this doc)** — confirm arc, chart list, and the three decisions above before building anything.
2. **Data prep script** — new `scripts/build_story_data.py` (or `energy_forecasting/story/` module, tbd on placement) computing the ~6 static JSON files into `deploy/story/data/`. Small, one-off aggregation, no new pipeline stage.
3. **Chart prototyping** — build each Plotly chart standalone (scratch HTML), checked against the dataviz skill conventions, before wiring into the scroll shell. This is the "some charts to tell the story" step the user asked for next.
4. **Scrollytelling shell** — HTML/CSS sticky-panel layout + Scrollama wiring + tab-jump nav, proven out on chapter 2 first.
5. **Wire remaining chapters** into the shell.
6. **Copy pass** — narrative text per chapter, DE/EN, user review pass on tone.
7. **Cross-link** — nav link from `deploy/index.html` to `deploy/story/index.html` and back; decide landing behaviour (does `deploy/` root become a chooser, or does the story site become the new root and the dashboard moves to `/dashboard/`? — open question, see below).
8. **Responsive/accessibility QA** — mobile layout, `prefers-reduced-motion` fallback (disable scroll-jacking, fall back to static stacked sections), keyboard nav for the tab bar.
9. **`make story-data` target + Playwright smoke test** (mirrors `test_dashboard.js`), deploy, compare against the reference site.

**Milestone:** story site live at `deploy/story/`, all chart list (A-E) rendering against real data, both scroll and tab navigation working, mobile-responsive, cross-linked with the existing dashboard.

## Decisions confirmed with user (2026-07-06)

- `deploy/story/` stays a subpath under the existing dashboard root (not a new front door). Step 7 cross-links `deploy/index.html` ↔ `deploy/story/index.html`.
- Scrollama.js (CDN) for scroll mechanics, confirmed over hand-rolled `IntersectionObserver`.
- Proceeding straight to chart prototyping (step 3) next.
