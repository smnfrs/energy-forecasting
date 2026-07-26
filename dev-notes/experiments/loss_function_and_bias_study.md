# Loss function & forecast-bias study

Status: **Phase 1 complete** (2026-07-24). Owner: Simon. Phase 1 results in §5a;
Phase 2 not started.

Goal: understand and reduce the systematic hour-of-day bias in the day-ahead
price forecast — especially the evening-peak under-prediction — by experimenting
with training loss functions, loss-diverse ensembles, and post-hoc debiasing.
Cast a wide net first (cheap single-model sweep), then pursue the promising
avenues.

---

## 1. Motivation & what the existing data already shows

The monitoring dashboard's hour-of-day chart shows a negative mean error (ME =
forecast − actual) with the worst bias and worst accuracy in the evening. That
chart is built from only ~5 days of production history, so before designing
anything we verified the pattern on data we already have.

**Data used:** the eight production ensemble members'
`mlruns/*/<run_id>/artifacts/predictions/oof_predictions.parquet`, combined with
the production inverse-MAE weights into a weighted-ensemble OOF series.
Leak-free CV out-of-fold predictions, **98,640 hourly rows, 2015-01-21 →
2026-04-22**. This is a proxy for the production approach (same models, same
losses, same weights) — not the exact monthly refit, but the best long-history
evidence available.

### Findings

**(a) The evening peak is robustly the worst-forecast hour.** Hour 19 UTC — the
evening ramp and literally the highest mean-price hour — has the worst MAE, RMSE
and most-negative ME in both the full 11 years and the recent 2 years. Not a
small-sample artefact.

**(b) "Negative bias at all hours" is regime-specific, not structural.**
Full-history ME is negative nearly everywhere, but that is dominated by the
2021–22 energy crisis (2022 evening ME ≈ −98 €/MWh). In the **recent 2 years the
overall ME ≈ 0**. What persists today is a *diurnal bias shape*, not a uniform
offset:

| Hours (UTC) | Recent ME (€/MWh) | Reading |
|---|---|---|
| Midday 10–14 | **+5 to +6** | over-predict — solar → low/negative prices, downside skew |
| Evening 17–20 | **−3 to −8** | under-predict — scarcity → upside spikes, right skew |

The sign of the bias tracks the diurnal **skew** direction, exactly as
median-targeting (MAE) theory predicts: MAE fits the conditional median, which
sits below the mean for a right-skewed (evening) distribution and above it for a
left-skewed (solar midday) one.

**(c) The loss function is a demonstrable driver — visible in our own data.**
Per-member recent-2y bias, grouped by training loss:

| Loss family | Midday ME | Evening ME |
|---|---|---|
| MAE trees (LGBM/XGB/CatBoost, 6 members) | **+8.3** | **−10.0** |
| Squared-error linear (Ridge/Lasso, 2 members) | **−4.7** | **+10.4** |

The squared-error members carry the **opposite** bias sign. Caveat: loss is
confounded with model class here (linear vs tree), so experiments must vary loss
*within a fixed tree model* to isolate the loss effect. Still, this is strong
prior evidence that mean-targeting or bias-aware losses will move the bias.

**(d) In the evening, bias is a small share of the error — mostly variance.**
Recent 2y, hour 19: bias² / MSE ≈ **3.5%**. A perfect debias removes ~3.5% of
evening MSE (~2% of evening RMSE). **Debiasing helps calibration and the
dashboard ME chart, but cannot materially fix evening accuracy** — that requires
variance reduction (better peak features / spike modelling), not recentring.

**(e) A flat static offset would be fragile.** Annual bias sign flips across
years (2015-16 +, 2018-19 −, 2021-22 hugely −, 2023 +, 2024-26 midday+/evening−),
and the evening bias is a **weekday** effect (recent evening ME −7.4 weekday vs
−1.7 weekend). Any correction must be rolling and conditional (hour × weekday ×
price level), applied strictly out-of-sample.

### Implications
- Loss function is the right primary lever; the data confirms it moves the bias.
- Mean-targeting (RMSE/Huber) or a bias-aware loss should flatten the diurnal
  bias and help RMSE, but likely costs a little MAE (median→mean trade). We do
  **not** pre-commit to a single metric — see §3.
- "Mixture of losses for diversity" is well-motivated: MAE and Huber tree
  members have offsetting evening biases → blending should flatten bias *and* add
  genuine diversity.
- **Ensemble-weighting trap:** production weights are inverse-MAE on a 90-day
  holdout, which structurally rewards the (biased) median members and will
  down-weight mean-targeting members back toward the biased blend. Any
  ensemble-level loss experiment must also revisit the weighting metric, or it
  self-sabotages.

---

## 2. Hypotheses under test

- **H1** The evening under-prediction is driven primarily by the MAE (median)
  objective interacting with right-skewed evening prices. → mean/Huber/bias-aware
  losses reduce it.
- **H2** No single loss dominates everywhere; a mixture of losses gives a better
  bias/accuracy trade-off than any single loss (diversity benefit).
- **H3** Post-hoc rolling conditional debiasing improves calibration (ME→0) and
  RMSE modestly, but does little for evening MAE (variance-bound, §1d).
- **H4** Modelling a variance-stabilised / skew-reduced target (log, asinh)
  reduces the diurnal bias at the root.

---

## 3. Metrics — report everything, decide afterwards

We do **not** pick one optimisation metric for the study. For each candidate we
log the full panel and choose after seeing the results:

- Overall **MAE**, **RMSE**, **ME**.
- **Per-hour** ME / MAE / RMSE (the primary diagnostic).
- **Evening block (17–20 UTC)** ME / MAE / RMSE — headline slice.
- **Midday block (10–14 UTC)** ME / MAE / RMSE — the opposite-skew check.
- **Pinball loss** at the evaluated quantiles (for quantile/expectile candidates).
- **PI coverage & width** — changing the point forecast shifts conformal
  calibration; confirm coverage stays ≈ 90%.
- Weekday/weekend split of the evening block.
- **Ensemble diversity & marginal value** — *not just standalone accuracy*. A
  candidate that is worse in isolation can improve the ensemble by decorrelating
  its errors (error diversity is half of the ambiguity decomposition:
  `ensemble_error = mean_member_error − diversity`). For every candidate log:
  - pairwise **error correlation** with the current production members (lower =
    more diverse), especially in the evening block where biases currently align;
  - **marginal contribution** — holdout MAE/RMSE/ME of the current ensemble *with
    the candidate added* (refit weights), vs without. This is the number that
    decides whether a "worse" loss earns a seat.

  A mean-targeting loss with opposite evening bias (cf. §1c, where the sq-error
  members bias +10 against the MAE trees' −10) is exactly the kind of member that
  can look mediocre alone yet flatten the blended bias.

The only places a scalar is unavoidable are internal to each candidate, and are
handled so the bakeoff stays fair:
- **Training objective** — the experiment's independent variable (each candidate
  defines its own).
- **Incidental hyperparameter tuning** (learning rate, depth, regularisation) and
  **early stopping** — tuned on **each candidate's own loss** on the validation
  window, so every loss gets its best shot. No shared global objective is imposed.

---

## 4. Data splitting — reuse the repo's existing scheme

**Do not invent a new split.** The price pipeline already has a canonical,
day-boundary-aware time-series split, and the study uses it verbatim so results
are comparable with production. The pieces (`config/modeling.py`,
`modeling/cv.py`, `modeling/price.py`):

- **Holdout:** the last **`HOLDOUT_DAYS = 90`** full days, carved out *before* CV
  via `carve_holdout()`. CV never sees it. Production uses this window to fit the
  ensemble inverse-MAE weights, calibrate conformal intervals, and report ensemble
  metrics.
- **CV:** `TimeSeriesSplitter`, **expanding** mode, day-boundary aligned
  (`SEARCH_CV_FOLDS = 3` during Optuna search, `VALIDATION_CV_FOLDS = 5` for the
  final retrain that captures OOF predictions over the whole pre-holdout pool).
- **Sample weighting:** exponential time-decay, `DEFAULT_WEIGHT_HALF_LIFE = 730`
  days (per-model tuned, e.g. 1095 for some production winners).
- **Candidate selection (relevant to the weighting question):** the ensemble
  category floor already keeps **best-OOF-MAE *and* best-OOF-RMSE per family**
  (`ENSEMBLE_FINAL_PER_CATEGORY = 2`), so RMSE-optimal members already get a seat
  — the current all-MAE ensemble is an outcome of inverse-MAE *weighting*, not of
  selection excluding them.

Two evaluation surfaces fall straight out of this, and we use both:

1. **Per-hour / evening diagnostics → the 5-fold OOF** (the exact artefact §1 was
   built from: leak-free, spans the whole pool, ~11 years / recent-2y slices).
   Long enough that rare evening spikes give stable per-hour ME/MAE/RMSE. Restrict
   *reporting* to the recent regime (post ~2023) for relevance, even though
   training uses full history with time-decay.
2. **Final head-to-head + weighting + conformal → the 90-day holdout**, exactly as
   production does.

Every candidate loss is trained through this same machinery (same
`TimeSeriesSplitter`, same 90-day carve, same 5-fold OOF capture, same weighting),
logged under **one MLflow experiment** so the comparison is apples-to-apples.

**Caveat to keep in view:** the 90-day holdout is short for *evening-spike*
stability (~90 evening blocks, few spikes). That is exactly why the per-hour and
evening diagnostics lean on the long OOF, while the 90-day holdout stays the
production-faithful final comparator. This is the known CV-vs-holdout tension in
this repo, not a reason to bolt on a bespoke walk-forward.

---

## 5. Phase 1 — wide net: single-model loss sweep

Cheap, high-signal. For each tree family (LGBM, XGB, CatBoost), retrain the
current best config changing **only** the objective, isolating the loss effect
within a fixed model class:

- **MAE** — baseline (current production).
- **RMSE / L2** — mean-targeting.
- **Huber** — δ grid, e.g. {50, 100, 200}. Mean-like but spike-robust.
- **Quantile / pinball** — τ ∈ {0.5, 0.6, 0.7} (upper quantiles bias toward the
  evening tail).
- **LogCosh / Fair** — smooth-MAE variants.
- **Custom bias-aware loss** — `MAE + λ·(mean residual)²`, or convex blend
  `α·MAE + (1−α)·MSE`; grid over λ / α. Direct expression of "a loss that
  includes bias".

**Baseline to beat, not a candidate:** rolling conditional **post-hoc debias** of
the plain MAE model (hour × weekday, shrunk, out-of-sample). Establishes the
ceiling of the free fix (expected small per §1d).

Deliverable: the §3 metric panel per candidate — including the **diversity /
marginal-value** columns — so we judge each loss on two axes: how it flattens the
diurnal bias standalone, *and* whether it decorrelates from the current MAE
members. Carry forward to Phase 2 both the best-standalone candidates and the
most-*complementary* ones, even if the latter are mediocre alone. ~1 day of
compute.

---

## 5a. Phase 1 results (2026-07-24)

Run by `scripts/phase1_loss_sweep.py` (29 candidates + Ridge/Lasso reference,
19 min on the tower). Each tree family used its production best-MAE config and
dataset (LGBM/XGB `fs_shap_top99`, CatBoost `fs_shap_top90`), varying only the
objective, through the exact repo split (`carve_holdout(90)` →
`TimeSeriesSplitter(5-fold, expanding)` OOF). Raw artefacts in
`data/experiments/phase1/` (`summary.csv`, `marginal.csv`, `per_hour.parquet`,
OOF/holdout predictions).

**Metric convention:** `ME = actual − forecast` (repo `calculate_metrics`), so
**+ve ME = under-prediction** — the *opposite sign* to §1's dashboard convention
(forecast − actual). Evening = 17–20 UTC, recent-2y OOF slice.

### Headline table (recent-2y OOF, per family, sorted by overall MAE)

| Family | Loss | all MAE | all RMSE | evening ME | evening MAE | midday ME |
|---|---|---|---|---|---|---|
| LGBM | mae (baseline) | **14.97** | 28.11 | **+7.48** | 20.69 | −8.03 |
| LGBM | quantile_t60 | 15.16 | 28.02 | +5.20 | 20.59 | −10.08 |
| LGBM | huberbias_l20 | 15.43 | 27.65 | **+2.41** | 20.97 | −6.38 |
| LGBM | blend_a50 | 15.48 | **27.48** | −0.06 | 20.68 | −7.04 |
| LGBM | rmse | 15.79 | 27.76 | −1.44 | 20.89 | −8.11 |
| XGB | mae (baseline) | **14.77** | 27.82 | **+7.19** | 20.54 | −9.07 |
| XGB | huberbias_l05 | 15.07 | 27.40 | +2.10 | 20.13 | −9.07 |
| XGB | huberbias_l10 | 15.08 | **27.30** | **+1.60** | 20.18 | −9.35 |
| XGB | blend_a50 | 15.13 | 27.87 | +3.07 | 20.47 | −7.66 |
| XGB | rmse | 15.38 | 28.05 | +1.63 | 20.42 | −7.76 |
| CatBoost | huber_d200 | 15.14 | 28.16 | +7.36 | 21.10 | −6.30 |
| CatBoost | rmse | 15.33 | 27.93 | +6.68 | 20.96 | −6.26 |
| CatBoost | mae (baseline) | 15.94 | 28.78 | **+13.23** | 23.27 | −5.77 |
| CatBoost | huber_d50 | 26.10 | 47.62 | — | 39.49 | — |

### Findings

**(1) H1 confirmed — the evening under-prediction is a median/MAE artefact.**
Across all three families the MAE objective gives the worst evening under-
prediction (+7 to +13). Mean-targeting (RMSE), Huber, and the bias-aware losses
collapse evening ME toward zero. The effect is large and consistent — loss is
unambiguously *the* lever on the diurnal bias.

**(2) §1d confirmed — evening error is variance-bound, not bias-bound.** Evening
**MAE/RMSE barely move** (~20.5–21.4) no matter the loss, while evening **ME**
swings from +13 to 0. Loss changes fix *calibration/bias*; they do **not**
materially improve evening accuracy. Real evening-accuracy gains still need
features (see `future_work_and_feature_ideas.md`). Expectations were correct.

**(3) Custom bias-aware losses are the best single models.** `huberbias`
(Huber core + λ·(mean residual)² penalty) sits best on the MAE-vs-bias frontier:
**XGB huberbias_l05/l10 cost only +0.3 MAE, give the best RMSE of any XGB loss,
and cut evening ME from +7.2 to ~+1.6–2.1.** LGBM huberbias_l20 / blend_a50 are
the LGBM sweet spot (evening ME +2.4/−0.1, RMSE *improved* vs MAE). Plain RMSE
also flattens bias but costs more MAE than huberbias for the same bias fix.

**(4) The MAE↔bias trade is mild and RMSE often improves.** Killing the evening
bias costs ≈0.3–0.8 overall MAE but frequently *lowers* overall RMSE (LGBM
28.11→27.48; XGB 27.82→27.30). So the "median vs mean" cost is small and partly
offset — good news for a blended ensemble.

**(5) CatBoost is the weakest family.** Its MAE is both worst-accuracy (15.94)
and worst-bias (+13.2); `Quantile:τ=0.7` over-corrects (MAE 18.4); and
`Huber:delta=50` is **numerically broken** (MAE 26, evening MAE 39) — a δ that
small is mis-scaled for CatBoost's internal target handling. Only CatBoost
RMSE / Huber:δ=200 are usable, and neither fixes its evening bias. Do not spend
Phase 2 effort tuning CatBoost losses; keep CatBoost-MAE as a diversity member
only.

**(6) Diversity / marginal value (H2).** Error-correlations with the incumbent
MAE-tree bloc are uniformly high (~0.93–0.95) because every candidate shares the
same features/data — loss alone does not buy much decorrelation. The bias-aware
LGBM losses are *marginally* the most diverse (huberbias_l10/l20 err_corr 0.928,
lowest of any non-degenerate candidate). When added to the 5-member incumbent
{LGBM/XGB/CatBoost-MAE + Ridge + Lasso} and inverse-MAE-reweighted on the 90-day
tail, **every non-broken candidate improves ensemble MAE** (Δmae −0.15 to −0.39;
biggest: LGBM blend_a50 −0.39, CatBoost rmse −0.36, LGBM huber_d50 −0.32). The
lone genuinely-decorrelated candidate, CatBoost huber_d50 (err_corr 0.54), is the
broken one and it **hurts** the ensemble (Δmae +0.49) — the textbook
"diverse-but-too-weak" member the study was designed to catch.

### Carry-forward to Phase 2

- **Best-standalone bias-aware members:** `XGB__huberbias_l10`, `XGB__huberbias_l05`,
  `LGBM__huberbias_l20`, `LGBM__blend_a50`. Small MAE cost, RMSE gain, near-zero
  evening bias.
- **Mean-targeting comparators:** `XGB__rmse`, `LGBM__rmse`, `CatBoost__rmse`.
- **Most-complementary (marginal + lowest corr):** `LGBM__blend_a50`,
  `LGBM__huberbias_l10/l20`, `CatBoost__rmse`.
- **Keep for MAE / diversity:** the family MAE members (they still own the best
  standalone MAE and anchor the low-error blend).
- **Drop:** CatBoost `Huber:δ=50` (broken); `quantile_t70` everywhere
  (over-corrects); CatBoost loss tuning generally.

The central Phase 2 question is now sharp: the bias-aware members flatten evening
bias but inverse-MAE weighting (which rewards the biased median members) will
fight them — so **the weighting metric, not just the member set, is the variable
to solve.** This is the §1 weighting trap, now empirically loaded.

---

## 6a. Phase 2 results — Tracks 1 & 2 (2026-07-25)

Run by `scripts/phase2_ensemble_weighting.py` (no retrain — pure post-processing
of the Phase 1 saved prediction columns). Artefacts in `data/experiments/phase2/`
(`results.csv`, `weights.txt`). **Honest walk-forward protocol:** weights fit on
the last 90 OOF days (2026-01-22 → 2026-04-22), evaluated on the disjoint 90-day
holdout (2026-04-23 → 2026-07-21) sitting immediately after. `ME = actual −
forecast` (+ve = under-predict). Evening = 17–20 UTC.

**Regime note:** the summer-2026 holdout is a *much harsher* regime than the
recent-2y OOF average — HOLD MAE ~17 / evening ME **+8.7** vs OOF-recent ~14 /
**+2.9**. The evening under-prediction is real and current, not a historical
artefact. All numbers below are the honest holdout eval.

### Track 1 — member set (weighting fixed = inverse-MAE)

| member set | MAE | RMSE | evening ME |
|---|---|---|---|
| incumbent (5, all-MAE) | 17.27 | 35.76 | **+8.73** |
| + LGBM blend_a50 (6) | 17.07 | 34.99 | +8.36 |
| + all 4 bias-aware (9) | 17.15 | **34.52** | **+7.69** |
| full bias-aware swap (5) | 17.97 | 34.47 | **+4.51** |

Adding bias-aware members to the incumbent is a *free, modest* win: RMSE −1.2,
evening ME −1.0, MAE flat. The **full swap** (evict the MAE trees) cuts evening
bias hardest (+4.51) but costs 0.7 MAE and over-corrects midday (+3.3) — keep the
MAE anchors, add diversity around them. **H2 confirmed but small**: loss-diverse
members earn a seat, but the decorrelation is limited (all share features/data,
err_corr 0.93–0.95 from §5a).

### Track 2 — weighting sweep (member set fixed = incumbent + 4 bias-aware, 9)

| weighting | MAE | RMSE | evening ME |
|---|---|---|---|
| inverse-MAE (production) | 17.15 | 34.52 | +7.69 |
| inverse-RMSE | 17.18 | 34.53 | +7.54 |
| simple average | 17.26 | 34.63 | +7.18 |
| MAE-optimal SLSQP (walk-fwd) | 17.25 | 34.44 | +7.35 |
| **bias-penalised γ≥0.5** | 17.25 | **33.92** | **+6.75** |
| MAE-optimal SLSQP **(in-sample)** | 16.66 | 34.89 | **+10.70** |

**The weighting trap, caught:** fitting MAE-optimal weights *in-sample* on the
holdout buys 0.6 MAE but **worsens** evening ME to +10.70 by loading the biased
median members — precisely the §1 self-sabotage. Walk-forward it's honest (17.25/
+7.35). The **bias-penalised** weight (minimise `MAE + γ·|evening ME|`) is the best
lever: evening ME +7.69→+6.75 and RMSE 34.52→33.92 at ~0.1 MAE cost. Penalty
saturates at γ≈0.5 (γ=0.5/1.0/2.0 give identical weights — the member pool caps
how much bias can be removed).

### Verdict on the no-retrain levers

Best no-retrain combo (9 members + bias-penalised weighting) vs production
incumbent: **MAE flat, RMSE −5.1%, evening ME −23% (+8.73 → +6.75)** — a genuine,
free calibration + RMSE improvement worth shipping. **But:**
- Evening **MAE** barely moves (22.48 → 22.02). **§1d re-confirmed on true
  out-of-sample data: the residual evening error is variance/feature-bound**, not
  reachable by loss choice or weighting. Real evening-accuracy gains need features
  (`future_work_and_feature_ideas.md`), not this study.
- The debias lever is structurally **weakest when needed most**: weights fit on the
  mild-bias OOF tail (evening ME ~+2–3) then applied to the severe-bias summer
  holdout (~+8). Regime shift outruns the weighting.

**Carry to promotion:** the 9-member set + bias-penalised weighting is the
candidate for the promotion gate (§6, requires fresh full-pipeline retrain +
conformal re-check). Track 3 (below) tests whether a variance-stabilised target
does better at the root; Track 4 stays low-priority (§1d).

---

## 6. Phase 2 — pursue the winners  (runbook for a fresh context)

**Start here.** Read §5a first (Phase 1 results + carry-forward list) and skim
`scripts/phase1_loss_sweep.py` (the reusable harness). Then work the tracks below.
**Tracks 1 & 2 are DONE — see §6a.** Remaining: Track 3 (retrain), Track 4 (low
priority).

### Reusable machinery & non-obvious gotchas (learned in Phase 1 — do not re-derive)
- **Predictions are already on disk.** `data/experiments/phase1/` (gitignored,
  73 MB) holds every candidate's leak-free OOF (`oof_predictions.parquet`, wide:
  `y_true` + one col per candidate) and 90-day-holdout preds
  (`holdout_predictions.parquet`), plus `summary.csv`, `marginal.csv`,
  `per_hour.parquet`, `incumbent.json`, and per-candidate `oof/*.parquet`,
  `hold/*.parquet`. **Tracks 1 & 2 below need NO retraining** — they are pure
  post-processing of these saved prediction columns. Only Tracks 3 & 4 train new
  models (reuse `_fit_predict_oof` / `_candidates` / `BASE` from the harness).
- **ME sign convention:** repo `calculate_metrics` gives `ME = actual − forecast`,
  so **+ve = under-prediction** — opposite to §1's dashboard sign. Keep straight.
- **Fold 1 = 48 rows** (2 days of Jan-2015; expanding 5-fold). Custom-objective
  trees can't fit it → the harness skips that fold (leaves NaN); those ancient
  rows are outside the recent-2y slice and dropped by any `dropna(how="any")`.
- **xgboost ≥2 custom objectives must apply `sample_weight` themselves** (LGBM
  applies it internally). The `_blend_obj`/`_huberbias_obj` factories already
  handle both via a `(y_true, y_pred, sample_weight=None)` signature.
- **Per-family fixed base** (dataset + hyperparams held constant, only objective
  varies): LGBM/XGB `fs_shap_top99`, CatBoost `fs_shap_top90`; XGB base carries
  `weight_half_life=1095`, the others `None`. See `BASE` in the harness.
- **Recent-2y reporting slice** = OOF end − 730 d (`≥ 2024-07-21`). Train on full
  history with time-decay; report on recent only (§4).

### Track 1 — Loss-diverse ensembles (H2) [no retrain]
Build ensembles from the saved OOF/holdout columns. Start from the incumbent
5-member set {LGBM/XGB/CatBoost-MAE + Ridge + Lasso} and add bias-aware members
(`XGB__huberbias_l10`, `LGBM__huberbias_l20`, `LGBM__blend_a50`) selected by
**marginal contribution** (already prototyped in `_marginal_screen`), not
standalone MAE. Core test of "a worse-alone model earns a seat via diversity."
Report the full §3 panel for each ensemble on the recent-2y OOF **and** the
production-faithful 90-day holdout — the diurnal-bias flattening is the headline,
not just overall MAE.

### Track 2 — Weighting-strategy sweep (the crux) [no retrain]
This is where Phase 1 pointed: inverse-MAE weighting rewards the biased median
members and will undo the bias-aware members' benefit. On a fixed diverse member
set, refit weights under: (a) inverse-MAE (production baseline), (b) inverse-RMSE,
(c) a bias-penalised score (e.g. minimise `MAE + γ·|evening ME|`), (d)
diversity-aware / SLSQP-constrained weights with a per-category floor. Compare the
**evening ME vs overall MAE vs RMSE** frontier. Fit weights on the recent 90-day
OOF tail (production's optimistic in-sample convention) but also report on a
held-back tail to check the optimism gap.

### Track 3 — Distribution shaping (H4) [retrain] — PARKED (machine contention)
Script written: `scripts/phase3_target_transform.py`. Holds loss = MAE (native)
and varies only `target_transform ∈ {none, log_shift, yeo_johnson}` for LGBM & XGB
(CatBoost dropped, §5a-5). Reuses `BASE`/`_panel` from the Phase 1 harness. Clean
H4 test: does a variance-stabilised target flatten the diurnal bias at the root,
independent of the loss lever.

**Status 2026-07-26:** first attempt aborted. The tower was running the Stage-2 LU
gen/load retrain (`logs/lu_retrain_rerun.sh`, `train gen-load --reuse-params
--parallel 4`) — load avg 33 on 16 cores, ~2× oversubscribed. Track 3 crawled
(one candidate in 2h43m vs ~40s unthrottled) AND was throttling the LU retrain
(~40×), so it was killed. The one completed row, `LGBM__mae__none`, reproduced the
Phase 1 baseline exactly (OOF MAE 14.97 / evening ME +7.48) — confirms the Track 3
harness is correct. **Re-run on a free machine** (est. ~5 min unthrottled), or
detached per the long-running-process protocol if the box is busy. Nothing else in
the study depends on Track 3.

### Track 4 — Evening specialist / hour-conditional loss [retrain]
Separate 17–20 UTC models, or a τ>0.5 expectile bump applied only to the evening
block. Lower priority given §1d (evening is variance-bound) — but the
hour-conditional weighting variant is cheap to try inside Track 2.

### Promotion gate
Whatever wins gets a **fresh full-pipeline retrain on the production split** (§4,
via `run_price_pipeline`, not this diagnostic harness) and a **conformal-coverage
re-check** (PI recalibrates on the changed point forecast) before any production
change. Nothing here touches `models/ensemble_config.json`.

---

## 7. Open questions / risks
- OOF here is the 5-fold validation CV, not the production monthly refit — good
  for direction, but Phase 2 winners must be confirmed with a fresh full-pipeline
  retrain on the production split (§4).
- Loss vs model-class confound (§1c) — keep loss variation within a fixed model.
- Conformal PI is calibrated on the point forecast; any winning change requires
  re-checking coverage before shipping.
- Evening accuracy is variance-bound (§1d) — manage expectations: loss changes
  fix *calibration/bias*, not the fundamental evening unpredictability. Real
  evening-accuracy gains likely need new features — see
  [`future_work_and_feature_ideas.md`](future_work_and_feature_ideas.md).
