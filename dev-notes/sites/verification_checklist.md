# Acceptance checklist — story/dashboard unification

Every problem raised at the start of this work, mapped to the stage that fixes it and
how to verify it. Walk this top-to-bottom at the end; tick each box only after seeing
it work on the deployed (or locally-served) surface. Companion to
[`unify_story_and_dashboard.md`](unify_story_and_dashboard.md).

Serve locally to check: `make serve-dashboard` (dashboard at `deploy/`), and open
`deploy/story/forecast/index.html` for the story deck. Do a full pass in **both light
and dark**.

Legend: **Stage** = where it's fixed · **Gate** = the stage's own verification gate.

---

## A. Unification & linking

- [ ] **A1. Story deck has a working light/dark toggle** (was dark-only).
  Stage 3. *Check:* toggle button in the story nav flips the whole deck; charts recolour;
  choice persists on reload (localStorage) and respects system default on first visit.
- [ ] **A2. Dashboard restyled to match the story site's visual system.**
  Stage 5. *Check:* dashboard uses the story design tokens/palette (not Bootstrap greys);
  looks of a piece with the deck; the same light/dark toggle works there too.
- [ ] **A3. Story deck is the main entry point.**
  Stage 3. *Check:* opening the site root lands on the story deck, not the dashboard.
- [ ] **A4. Prominent, larger link from the story to the dashboard**
  ("for those who only want to see the numbers"). Stage 4. *Check:* a clearly highlighted,
  larger-than-a-nav-link button to the live dashboard; wording points the "just the numbers"
  reader there.
- [ ] **A5. History site is NOT linked** (it isn't ready).
  Stage 3/4. *Check:* no "The history →" nav link and no closing "Read the history" CTA
  anywhere; grep finds no live link to `story/index.html`.

## B. Story deck — content & layout

- [ ] **B1. Hero/opening page looks good.**
  Stage 4 (layout) + Stage 6 (copy). *Check:* the first slide reads cleanly and has visual
  interest (a headline-stat treatment), not a wall of text.
- [ ] **B2. "volatile" no longer appears twice in one sentence.**
  Stage 6. *Check:* read the hero paragraph — the doubled word is gone.
- [ ] **B3. Hero heading is not too long.**
  Stage 6. *Check:* the H1 is a short, punchy line.
- [ ] **B4. Summary text makes sense** (was: "wind up 2.5 points and coal down 2.3 points").
  Stage 6. *Check:* each narrative box reads as a plain, correct sentence — shares expressed
  as "wind's share rose 2.5 pp to X%", no bare "points", no undefined/NaN.
- [ ] **B5. Charts fit one screen — no scrolling.**
  Stage 4. *Check:* on each slide the chapter head + chart + narrative + nav controls all fit
  within the viewport; nothing overflows or requires page scroll (test a laptop-height window).
- [ ] **B6. Actuals overlap the forecast** (they meet at the seam).
  Stage 4 (+ Stage 1). *Check:* on the gen/load forecast slide the recent-actual (grey) line
  connects to tomorrow's forecast (blue) with a genuine overlap window, no gap.
- [ ] **B7. Load forecast no longer dives to zero.**
  Stage 1 (root cause) + Stage 4. *Check:* the load panel stays at ~40–60 GW across the whole
  horizon; no collapse to 0 at the tail.
- [ ] **B8. The two-charts-on-one-slide page is split.**
  Stage 4. *Check:* price forecast and the "what's driving it" SHAP breakdown are on separate
  slides, each with its own dot/nav entry.
- [ ] **B9. Last (track-record) chart shows only the last 7 days.**
  Stage 4. *Check:* the track-record chart spans 7 delivery days, not the full history.
- [ ] **B10. Last-page text reworked.**
  Stage 6. *Check:* the track-record narrative reads clearly and factually.
- [ ] **B11. Tone is professional and factual throughout** — no overpromising or interpreting.
  Stage 6. *Check:* skim every narrative box; nothing editorialises or predicts beyond the numbers.

## C. Dashboard — content, data & panels

- [ ] **C1. Every chart has explanatory text.**
  Stage 6. *Check:* each chart on all three tabs has a one-line caption saying what it is.
- [ ] **C2. Load number no longer goes to zero** (properly diagnosed, not bodged).
  Stage 1. *Check:* national load actuals + forecast stay at realistic GW to the horizon end;
  root cause was the zero-fill in `aggregate_national()` + `min_count` under-summing, both fixed.
- [ ] **C3. Individual gen/load forecasts include Luxembourg.**
  Stage 2 (models) + config cards (done). *Check:* the wind-onshore and solar cards show a
  "Creos" sub-line (post-retrain), matching the load card.
- [ ] **C4. A chart exists for the 5th forecast series (`gen_load_diff`, the net balance).**
  Stage 5. *Check:* the Generation & Load tab has a `gen_load_diff` panel (small ± quantity,
  ~net exports + losses), forecast shown.
- [ ] **C5. Monitoring: data before 1 July 2026 is cut** (25 June was clearly broken).
  Stage 1. *Check:* `gen_load_errors_summary.json` and the monitoring chart start ≥ 2026-07-01.
- [ ] **C6. Monitoring: x-axis is 14 days.**
  Stage 1. *Check:* the monitoring/gen-load-error series shows ≤ 14 points.
- [ ] **C7. CV-MAE chart removed.**
  Stage 5 (agreed: remove). *Check:* the "linear-beats-trees" CV-MAE panel is gone from Monitoring.
- [ ] **C8. Gen/load accuracy chart is readable (last 14 days).**
  Stage 1 (server filter) + Stage 5 (label). *Check:* the gen/load accuracy chart shows ~14 days,
  labelled as such — not 30+ unreadable days; no client-side OOF-divider hack.
- [ ] **C9. Deployment feature audit cut.**
  Stage 5. *Check:* the unreadable feature-audit panel is gone; a labelled placeholder marks
  where the future data-collection flags panel will go.

## D. Meta-feedback on the process (already actioned)

- [x] **D1. Luxembourg checked against the EMA approach.** EMA models LU wind/solar per-region
  (`eu_locations.py` LU_CREOS); our merge had dropped it — decision: restore fidelity (Stage 2).
- [x] **D2. Work broken into stages with verification gates.** See the plan doc's Stages 1–7.
- [ ] **D3. Text generation delegated to a focused sub-agent.**
  Stage 6. *Check:* the copy (hero, narrative templates, captions, LU note) was produced by a
  sub-agent with a focused brief, then integrated — confirm at the Stage 6 copy review.

---

## Deferred (out of scope now — do NOT expect these)

- **Data-collection flags panel** (missing / forward-filled data) — replaces the cut feature
  audit; placeholder only for now. Track separately.
- **History scrollytelling page** (`deploy/story/index.html`) — unchanged and unlinked until ready.

## Cross-cutting release checks (Stage 7)

- [ ] Deployed JSON regenerated by a real inference run (panels render against fresh data).
- [ ] Both surfaces walked in light **and** dark.
- [ ] Full test suite green.
- [ ] Pushed to `origin/master` **before** prod picks up the Stage 2 models (deploy-skew caution).
