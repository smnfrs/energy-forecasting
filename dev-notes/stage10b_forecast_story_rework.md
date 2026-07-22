# Stage 10b: Forecast Story Rework

A second pass over the Stage 10 forecast story (`deploy/story/forecast/`) to fix
six problems the user identified after living with the deployed page, plus a few
found alongside them. Not a new stage — a revision of Stage 10, hence "10b".

**Detailed plan:** this document
**Status:** planned 2026-07-21. **Workstreams A + B implemented, committed, pushed
and deployed live 2026-07-22** (commit `5a8e339`; `deploy_static` run 29910815408).
Supersedes parts of Stage 10's frontend and narrative design; the Stage 10
SHAP/facts *numeric* pipeline is unchanged. **All workstreams A–E implemented
2026-07-22. A + B are committed/deployed; C, D, E implemented locally and not yet
committed (pending user review of the final version).**

**Progress:**
- **A (copy & claims) — done, live 2026-07-22.** Hero reframed to the two-cause
  throughline (renewable share *and* commodity prices → volatility → forecasting
  as enabler); `TODO(user)` and the "cheaper" claim removed. Year labels relabelled
  ("the last 12 months" / "the 12 months before" / "the same week a year ago").
- **B (template narrative, LLM dormant) — done, live 2026-07-22.** All four
  narrative boxes now built by deterministic JS sentence builders
  (`yearlyGenLoadSentence`, `yearlyPriceSentence`, `forecastGenLoadSentenceFrom`,
  `forecastPriceDriverSentence` in `forecast-story.js`) from already-published JSON;
  pure/DOM-free for testing. MW/MWh unit bug fixed at source. `stat-line-group` spans
  removed (no more double presentation). "AI summary" chrome → "Summary". Degradation
  contract met (verified via node harness: full data, missing JSON, absent fields —
  no `undefined`/`NaN`/`null %`, muted fallback when whole box degrades). Groq guarded
  off in both workflows (`daily_forecast.yml` `if: false`; `story_data.yml`
  yearly-narrative line removed); `narrative.py` + tests kept (8 pass). Frontend no
  longer fetches `narrative_*.json`.

- **C (declutter the two yearly charts) — implemented 2026-07-22, not yet
  committed.** Gen/load chart now weekly-resampled (`weeklyGenLoad`, ~53 points,
  7-day mean, "MWh/day" scale preserved); the load line is **removed from the
  default view** (its beat-reveal home lands in D) but weekly load means stay in
  the detail table. Price chart replaced the ~8,760-point hourly `scattergl` smear
  with a daily-mean line + shaded **daily min–max** band (`dailyPriceBands`, 365
  points). Both new functions pure/DOM-free and verified against the committed
  `facts_yearly.json` (53 weekly points no-null; 365 daily bands, 87 days with a
  negative hour, range −499→666, all finite). Chart copy/titles in `index.html`
  updated to match (no more "load overlaid" / "every hourly price"). **Decisions
  locked:** kept **all 7 fuel groups** (weekly resampling declutters enough; the
  coal-vs-renewables contrast is the thesis) and **min–max** band (shows the full
  swing envelope incl. negative troughs, vs p10–p90 which would hide them).

- **D (sticky-graphic scrollytelling) — implemented 2026-07-22, not yet
  committed.** Each of the four chart chapters is now a sticky graphic that pins
  while 2–3 short **beats** scroll past (`.beat[data-step]` under
  `.chapter-copy[data-scene]`). A lightweight **scene registry** in
  `forecast-story.js` (`SCENES` / `registerScene` / `applyBeat`, exposed on
  `window.SCENES`) drives each beat's chart state, idempotently (applyBeat sets
  the full state for a step, so entering from either scroll direction lands the
  same): gen/load — renewable-share → dim-to-wind+coal shift → load-line reveal;
  price — mean line → swing band fades in → deepest-trough annotation; gen/load
  forecast — solar-shape (authored) → annotate the biggest-deviation panel;
  price forecast — line → PI band fades in → highlight top SHAP bar. Computed beat
  slots (`data-fill`) are filled from the same JSON the charts use and verified
  against live snapshots (renew 48%, "wind up 2.5 / coal down 2.3 points", €93/MWh,
  475 negative hours, "wind offshore 19% below", "Neighbour prices +16.4 / Residual
  generation −8.6"). Transitions use the simplest mechanism that reads (Plotly
  `restyle` opacity/visibility, `relayout` annotations toggled) — no chained
  animations. **Arrow-key chapter paging removed** entirely; nav is now click-to-
  scroll + an IntersectionObserver active-chapter highlight. **Mobile (<860px):**
  Scrollama stepping disabled, beats hidden, the lead `.narrative-box` (the
  Workstream B templated summary) shown as the consolidated caption, and each chart
  left in its final annotated state (`registerScene` applies the last beat on
  mobile). Degradation contract preserved: a beat whose computed input is absent is
  hidden (`fillBeat(..., null)`), scenes that fail to build never register, and
  `plotted()` guards every restyle. Verified end-to-end with a headless node harness
  (stubbed `document`/`window`/`Plotly`/`fetch` over the real JSON): all 5 charts
  plot, all 4 scenes register, every `applyBeat` step runs without throwing; HTML
  tag-balanced (11 beats, 6 fill slots).

- **E (closing track-record chapter) — implemented 2026-07-22, not yet
  committed.** The bare "Want the longer story?" CTA block is replaced by a
  sticky-graphic chapter ("How good is any of this?", nav link added). Chart
  (`buildTrackRecordChart`): the issued price forecasts from
  `forecast_history.json` concatenated into one day-before track, realised prices
  from `actuals.json` overlaid on exactly those delivery dates, plus the PI band.
  Resolving beat pulls the rolling MAE from `errors_summary.json` ("Over the last
  7 days … off by about €10/MWh on average", RMSE €16 in the mobile summary via
  pure `trackRecordSentence`). The history cross-link survives as a `.closing-cta`
  that is a **sibling** of `.beats` (not a beat), so it stays visible on mobile
  where beats are hidden. Verified in the headless harness (6 charts plot; the
  track-record chart is static, so no scene is registered — beat highlighting
  still works and mobile shows the built/final state).

**Carry into C (and later):**
- The `.stat-line-group` CSS in `forecast-story.css` is now **dead** (spans removed
  in B). Leave for now; remove if C/D don't reintroduce stat lines.
- **Cross-file date alignment:** `forecastGenLoadSentenceFrom` filters gen/load
  forecasts to the price forecast's delivery day, falling back to the first 24
  forecast hours when snapshots don't line up (the committed repo snapshot hits the
  fallback; live aligned data uses delivery-day filtering). Re-check on live data.
- **No JS test harness landed yet.** The builders were factored pure/DOM-free
  precisely so a minimal node harness (per § Testing) can be stood up; the sanity
  check so far is the throwaway node script, not a committed test. Still an open item.
- Node lives at `/home/smnfrs/miniconda3/envs/nodetmp/bin/node` (no system `node`).

> Naming note: Stage 10's own doc uses "10a–10f" for its internal sub-stages
> (its "10b" is the yearly-recap narrative). To avoid collision, this doc's units
> of work are **Workstreams A–D**, not letters that clash with those sub-stages.

---

## Problems being addressed

From the user, after using <https://smnfrs.github.io/energy-forecasting/story/forecast/>:

1. **"Trailing year / prior year" language is confusing.**
2. **Much of the AI summary is unnecessary** — the same text could be prewritten
   with the calculated numbers slotted in.
3. **Much of the AI text makes no sense** — dense number-recitation; and a real
   unit bug (see below).
4. **The first two charts are too busy** to read anything from.
5. **Each "page" is taller than the browser viewport**, so arrow-key navigation
   between chapters works badly.
6. **The intro claim that "renewables make German power cheaper" is dubious** and
   the user does not want to rely on it.

Raised in review of this plan (another model, 2026-07-21) and folded in below:
- The plan specified the scroll *mechanism* but deferred the *beat copy* — the
  same defer-the-hard-part pattern as the `TODO(user)` hero it fixes. Beats are
  now drafted up front (§ "Draft beats").
- "Every beat pegs to a computed callout" was too strict — structural truths
  (solar → 0 every night) deserve fixed authored beats. Now allowed.
- The arc didn't close: the hero promises forecasting-as-enabler, but the page
  trailed off into a "read the history" link. A closing chapter now resolves it.
- The mobile fallback (all annotations at once) would re-create the clutter
  Workstream C removes. Mobile now shows few/no on-chart annotations.
- Templated sentences still break on missing/malformed fields; the empty/partial
  state is now specified and tested.

Found alongside:
- A literal `TODO(user)` placeholder comment is shipped live in the hero intro
  (`index.html:29-31`) — the intro copy was never finalised.
- **Unit bug in the live AI text:** the deployed forecast note reads *"mean load
  forecast is 55313.3 MWh"* and *"wind onshore … 9353.3 MWh"* — those are hourly
  **MW** values, not MWh. The Groq prompt hands the model bare numbers with no
  units, so it invents wrong ones. (`narrative_forecast.json`, live 2026-07-15.)
- **Double presentation:** the `stat-line-group` already prints fuel mix %, mean
  price, extremes and negative-hour count; the AI summary then recites the same
  numbers again next to it.
- **Chart 3 (gen/load forecast, 2×2)** has the inverse of the busy-ness problem:
  7 days of actuals with tomorrow's forecast a thin sliver at the right edge —
  the thing it exists to highlight is the least visible part.

---

## Decisions (settled with the user before writing this plan)

- **Year framing: relabel, keep rolling.** The windows stay rolling 12-month /
  prior 12-month / last-7-days / same-week-a-year-ago — statistically sounder
  than calendar+YTD (early in a calendar year, YTD is a tiny noisy sample). Only
  the *words* change. No change to `scripts/build_forecast_story_data.py`.
- **Kill the LLM on all four narrative boxes; replace with deterministic
  templated sentences** built in JS from the already-published facts. Fixes the
  unit bug and the nonsense at the root, and removes the double-presentation by
  folding the stat-line numbers into the one templated paragraph.
- **Leave the LLM code dormant, do not delete it.** `narrative.py`, its tests,
  and the existing `narrative_*.json` files stay in the repo. The workflow step
  that invokes it is guarded off; the frontend stops depending on it. Re-enabling
  later is a one-line workflow change plus a frontend toggle.
- **No LLM SHAP interpretation for now.** A template already states the top SHAP
  categories and their signed EUR/MWh contribution honestly. The only thing an
  LLM would add on top is *causal narrative* — precisely the invented causality
  the existing disclaimer disowns. "Genuine SHAP interpretation with real context
  feeds" is parked as a backlog item (see `dev-notes/roadmap.md`), not half-built
  here.
- **Sticky-graphic scrollytelling, keep Scrollama.** Scripted narrative beats
  fight *daily-changing* data, so storytelling here is **data-driven**: compute
  the notable fact each day and draw it on the chart. Each chart pins (sticky)
  while 2–4 short text beats scroll past, each beat revealing a computed
  annotation / chart-state change. Keeps library + feel continuity with the
  Stage 9 historical site.
- **Remove arrow-key chapter paging.** It was fighting oversized chapters; the
  sticky-graphic model wants natural scrolling (chart stays, text moves), which
  dissolves problem #5 rather than working around it.
- **Mobile: degrade, don't drop.** Desktop (≥ ~860px) = sticky chart + scrolling
  beats + step-revealed annotations. Mobile (< 860px) = drop sticky/step; render
  each chart once in its **final fully-annotated state** with consolidated text
  below it (normal figure-and-caption). The existing `@media (max-width: 860px)`
  block in `story.css` already stacks the grid, makes `.chapter-sticky` static and
  puts the chart first (`order: -1`) — we extend that, not rebuild it. Explicitly
  **avoid** floating translucent text cards over charts (wrecks data legibility).

---

## Workstream A — Copy & claims (no pipeline change)

Files: `deploy/story/forecast/index.html`.

1. **Hero reframe.** Delete the `TODO(user)` comment block (`index.html:29-31`)
   and the "cheaper" claim in the H1. Adopt the user-authored throughline (this is
   the spine the whole page serves, not just the hero):

   > German energy generation needs to adapt for the 21st century, but renewables
   > create challenges. The increasing proportion of renewables in the energy mix,
   > together with more volatile commodity prices, has caused the volatility of
   > electricity prices to explode. To cope with this volatility, we need a
   > flexible energy system, and that flexibility requires high-quality forecasts.

   Why this over the current draft: it names **two** causes (renewable share *and*
   commodity prices), which avoids the "renewables are the reason / renewables make
   power cheaper" overclaim; and it ends on **forecasting as the enabler of
   flexibility**, giving the page a destination to arrive at. Do not assert net
   consumer-price effects (merit-order net impact is genuinely contested).
   The **closing chapter (Workstream E)** must return to this promise — the page
   arrives at "here is tomorrow's forecast and how far to trust it", not a
   trail-off.
2. **Relabel year comparisons** everywhere in copy and templated text:
   - "trailing year" → "the last 12 months"
   - "prior year" → "the 12 months before"
   - "same 7 days prior year" → "the same week a year ago"
   Chapter kickers "The last year" → "The last 12 months".

Independently shippable; lowest risk.

## Workstream B — Template the narrative, LLM dormant

Files: `deploy/story/forecast/forecast-story.js`, `common.js`,
`forecast-story.css`, `index.html`, `.github/workflows/story_data.yml`,
`.github/workflows/daily_forecast.yml` (whichever invokes narrative),
`energy_forecasting/deploy/narrative.py` (guard only, no deletion).

3. **Templated yearly summaries.** Replace `renderNarrative(...)` for
   `narrative-gen-load-yearly` and `narrative-price-yearly` with deterministic
   sentence builders that read `facts_yearly.json` (already fetched for the
   charts — reuse, don't refetch). Correct units. Fold the `stat-line-group`
   numbers into this paragraph so nothing is shown twice; remove the now-redundant
   `<span id="stat-*">` lines from those two chapters.
4. **Templated forecast summaries.** Same for `narrative-gen-load-forecast`
   (compare tomorrow's gen/load to the last 7-day mean; flag any series near its
   recent min/max) and `narrative-price-driver` (top 2–3 SHAP categories by
   |mean signed contribution|, each with sign and EUR/MWh, read from
   `price_shap.json`). **Fix the MW/MWh unit bug** — load and gen are MW. Keep the
   existing SHAP epistemic-humility disclaimer sentence verbatim.
5. **Retire the "AI summary" chrome.** Drop or rename the `.narrative-box::before`
   "AI summary" label (→ "Summary" or remove). Keep the graceful-fallback styling
   for the (now unused) unavailable state in case the LLM is re-enabled.
6. **Guard the LLM invocation.** Remove/`if: false`-guard the narrative step in
   the story-data workflow so Groq is no longer called on schedule. Leave
   `narrative.py`, `test_deploy_narrative.py` and the committed `narrative_*.json`
   in place. Add a one-line note at the guard pointing here.

6a. **Write sentences, not slot-fills.** "Wind 45%, solar 12%, gas 18%" is
   correct but robotic — that trades wrong-but-fluent LLM text for
   right-but-wooden template text. Each builder emits a sentence a person would
   actually say, with the numbers embedded in prose and only the leading
   one or two facts surfaced (not every field dumped).
6b. **Degradation contract (missing/partial data).** A deterministic builder
   still breaks on an absent field or malformed JSON — and the page's stated
   promise is that the charts are always live regardless. Each builder must:
   - never throw (guard every field access; the whole box, not the page, is what
     degrades);
   - if the underlying JSON is missing/unparseable → show the existing muted
     "summary unavailable — the numbers above are still live" fallback;
   - if *some* fields are present → emit only the sentence(s) whose inputs exist,
     dropping clauses whose inputs are absent rather than printing `undefined`,
     `NaN`, or `null %`.
   Specify per builder which fields are load-bearing vs. droppable.

Decision: whether the frontend should stop *fetching* `narrative_*.json` entirely
(cleanest) vs. fetch-but-ignore. Recommend **stop fetching** — the templated text
is the source of truth; dead fetches only invite confusion.

## Workstream C — Declutter the two yearly charts

Files: `deploy/story/forecast/forecast-story.js`. Data source unchanged
(`facts_yearly.json` already carries the daily/hourly series).

7. **Chart 1 — yearly generation & load.** The current daily stack of 7 fuel
   groups + a load line over 365 points is a hairball, and the generation-stack /
   load-line overlay invites the misread that the gap is imports.
   - Resample the daily gen series to **weekly** (~52 points) in JS before
     plotting (facts carry `gl.date` + per-group daily arrays; bucket by ISO week
     or 7-day blocks and sum/mean).
   - **Load line decided (not vague):** removed from the default view, but given
     a home — it fades in as a revealed line during **one dedicated beat** of this
     chapter ("demand barely moves next to supply"). So load context is kept, just
     not shown by default where it invites the gap-is-imports misread.
   - Consider collapsing to fewer fuel groups if still busy (e.g. wind, solar,
     other-renewable, gas, coal, other) — tune visually, keep the Stage 9 colour
     tokens.
8. **Chart 2 — yearly prices.** Replace the ~8,760 hourly `scattergl` points with
   a **daily-mean line + shaded daily min–max band** (or p10–p90). Shows both the
   level *and* the volatility — the page's whole thesis — instead of a smear.
   Compute daily aggregates in JS from `price.timestamp` / `price.price`.

## Workstream D — Sticky-graphic scrollytelling

Files: `index.html`, `forecast-story.js`, the inline nav/scroll `<script>`,
`story.css` / `forecast-story.css`.

9. **Beats first, engine second.** Author the beats (§ "Draft beats") *before*
   building the scroll engine — the number of steps and what each reveals falls
   out of the beats, not the reverse. Then keep the existing `.scroll-wrap` grid
   (copy | sticky chart) and make each chart genuinely sticky through its 2–4
   stacked beats. Reduce chart heights/padding so the sticky graphic sits
   comfortably in the viewport beneath the nav.
10. **Scroll engine.** Keep Scrollama. Set up steps at the *beat* granularity
    (not the chapter granularity). `onStepEnter` drives a per-beat chart state:
    reveal/focus transition (line draws, band fades in after the line, SHAP top
    bar highlights) + the beat's computed annotation.
10a. **Transition budget.** Sequencing a line-draw → band-fade → bar-highlight per
    beat means chained `restyle`/`relayout`/`animate` calls that can stutter.
    Budget real time in Workstream D for this, and prefer the *simplest* mechanism
    that reads (opacity via `restyle`, static annotations toggled on) over full
    Plotly transitions where the two look the same.

11. **Annotations — computed *and* authored.** Not every beat pegs to a daily
    number. Structural truths (solar → 0 every night; demand barely moves) are
    fixed authored beats, written final now. The **computed** annotations below
    are deterministic, from data already present, so they refresh daily with no
    re-authoring:
    - Yearly gen/load → YoY shift callout ("wind +2.5pts, coal −2.3pts").
    - Yearly price → mark negative-price troughs; band the most-expensive hour.
    - Forecast gen/load → auto-flag whichever series is most unusual vs its 7-day
      range ("load ~7% above this week's average").
    - Price forecast → mark tomorrow's peak hour; wide band → "more uncertain".
    - SHAP → highlight the top driver bar; echo it in the templated sentence.
    Each annotation is the *same* computed fact the templated sentence uses, so
    text and chart reinforce.
12. **Remove arrow-key paging.** Delete the `keydown` chapter-jump handler and the
    `jumpTo`/`CHAPTERS` paging logic. Keep nav-link click-to-scroll and active-
    section highlighting (Scrollama `onStepEnter` → nearest chapter).
13. **Mobile fallback.** Below 860px, disable Scrollama stepping and render each
    chart once with the consolidated templated text below it, reusing the existing
    `@media (max-width: 860px)` static-stack rules. **Do not dump every on-chart
    annotation at once** — that re-creates the clutter Workstream C removes, on the
    device least able to cope. Show at most the single most important annotation
    per chart (or none) and let the text below carry the rest. No text-over-chart
    overlays.

## Workstream E — Closing chapter: forecast track record

**Approved in scope 2026-07-22.** Beyond the original six issues, but pulled in
because it (a) supplies the evidence the reframed hero implicitly promises
("forecasting is what lets us cope") and (b) closes the arc.

Data is already published — no pipeline work:
- `deploy/data/forecast_history.json` — past *issued* price forecasts with PI
  bands, keyed by `issued_at` (currently `target: price`, list of daily issues).
- `deploy/data/actuals.json` — realised hourly prices to overlay against them.
- `deploy/data/errors_summary.json` — `dates` / `mae` / `rmse`, last 7 days
  (rolling accuracy; extend the window later if wanted).

14. **New closing chapter** ("How good is any of this?") replacing the current
    bare "Want the longer story?" CTA block (`index.html:140-148`). Sticky chart:
    recent issued forecasts vs. realised actuals (or a rolling-MAE line), with the
    headline "off by ~[MAE] €/MWh on average over the last [N] days" as the
    resolving beat. Then the existing CTA to the history page as the final beat,
    so the cross-link survives.

## Draft beats

Beat *intent* + number slots, drafted now so the engine's step count is derived
from them. `[…]` = computed slot (final wording locks when real values render);
unbracketed = fixed authored copy, final now. Tag: **(A)** authored/structural,
**(C)** computed.

**Ch. Generation & load** — sticky: weekly gen stack
- (C) Over the last 12 months, wind and solar together supplied **[X]%** of
  Germany's electricity. *(full stack shown)*
- (C) That's shifting fast — versus the 12 months before, wind **[+Δpts]** and
  coal **[−Δpts]**. *(highlight those bands)*
- (A) Demand, by contrast, barely moves. *(load line fades in — its one home)*

**Ch. Prices** — sticky: daily-mean line + min–max band
- (C) Day-ahead prices averaged **[X] €/MWh** over the last 12 months. *(line only)*
- (A) But the average hides the real story: the swings. *(min–max band fades in)*
- (C) Prices even went **negative in [N] hours** — moments with more supply than
  the grid could use. *(mark troughs)*

**Ch. Gen/load forecast** — sticky: tomorrow vs last 7 days
- (A) Solar collapses to zero every night and peaks at midday — the forecast has
  to get that shape right, every day.
- (C) For tomorrow, **[series]** stands out: **[X]% [above/below]** this week's
  average. *(flag it; if nothing is unusual, say so)*

**Ch. Price forecast** — sticky: forecast + PI band, then SHAP bars
- (A) Here's what the model expects tomorrow's price to do. *(line draws)*
- (C) The shaded band is its uncertainty — **[wider today / typical]** means
  **[less / normally]** sure. *(band fades in)*
- (C) It leans most on **[top category] ([±N] €/MWh)**, then **[2nd] ([±N])**.
  This is the model's own reasoning, not proven real-world cause. *(highlight top
  SHAP bar; disclaimer kept verbatim)*

**Ch. Track record (Workstream E)** — sticky: issued forecasts vs actuals
- (C) So how good is any of this? Over the last **[N] days** the price forecast
  was off by **[MAE] €/MWh** on average. *(evidence)*
- (A) Good enough to plan around — which is the whole point. → *read the history.*

---

## Sequencing

A → B → C → D → E. A and B are low-risk and independently shippable. C is visual
tuning on top. D is the largest change and restyles the shell for everything, so
it goes over final content. E (if approved) is the new closing chapter, built on
D's engine. Each workstream ends deployable.

## Testing

- `test_deploy_narrative.py` stays green (code dormant, not deleted). If the
  frontend stops fetching `narrative_*.json`, no Python test asserts that fetch.
- No new Python behaviour — Workstreams are frontend + workflow-config.
- **Degradation cases (per § 6b) must be covered, not just the happy path:**
  each templated builder tested against (i) full data, (ii) the JSON missing
  entirely, (iii) individual load-bearing fields absent/`null`/`NaN`. Assert no
  `undefined`/`NaN`/`null %` ever reaches the DOM and the box degrades to the muted
  fallback when it should.
  - **Caveat: the repo has no JS test harness today** (all tests are pytest; the
    frontend is untested). Options: (a) stand up a minimal harness (e.g. a tiny
    node script importing the pure builder functions with fixture JSON — feasible
    if builders are factored to not touch the DOM), or (b) document these as
    explicit manual cases with saved fixture files. **Sub-decision for the user.**
    Recommend (a) for the pure sentence builders — they're the fragile part and
    cheap to unit-test once separated from rendering.
- Manual: serve `deploy/` locally, check each chart/beat against the committed
  `facts_yearly.json` / `price_shap.json` / `forecast_history.json` snapshots,
  desktop and < 860px. Confirm the guarded workflow no longer calls Groq (read
  the diff / dry-run).

## Open items to confirm during implementation

- **JS test harness** — default (set 2026-07-22): stand up a minimal DOM-free
  node harness for the sentence builders. Revisit only if factoring the builders
  out of rendering proves disproportionate.
- ~~Chart 1 fuel-group collapse: 7 groups weekly vs. a reduced set~~ — **decided
  2026-07-22: kept all 7 groups.** Weekly resampling alone declutters the hairball.
- ~~Chart 2 band: min–max vs. p10–p90~~ — **decided 2026-07-22: min–max**, to keep
  the negative-price troughs the thesis relies on.
- Final wording of the **computed** beat slots — locks when real values render.
