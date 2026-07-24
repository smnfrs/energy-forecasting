# Loss function & forecast-bias study

Status: **planned** (2026-07-24). Owner: Simon. No compute run yet.

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

## 6. Phase 2 — pursue the winners

- **Loss-diverse ensembles (H2):** build ensembles from mixtures of losses
  (best-MAE + best-Huber/RMSE + complementary candidates from Phase 1), selecting
  members by **marginal contribution**, not standalone MAE. This is the core test
  of "a worse model can help the ensemble via diversity."
- **Weighting-strategy sweep:** refit the ensemble under (a) inverse-MAE (current),
  (b) inverse-RMSE, (c) a bias-penalised score, and (d) diversity-aware /
  constrained weights. The current train-on-MAE / weight-on-MAE choice was
  inherited from EP because there it won on MAE, ME *and* RMSE simultaneously —
  an easy call. Here the objectives may genuinely trade off, so the weighting
  metric is itself a variable, not a given. Addresses the weighting trap in §1.
- **Distribution shaping (H4):** model `asinh` / log-transformed price, back-
  transform, re-measure the median↔mean gap and diurnal bias.
- **Evening specialist / hour-conditional loss:** separate peak-hour models, or a
  τ>0.5 expectile bump applied only to 17–20 UTC.
- Whatever clears Phase 1 gets a fresh full-pipeline retrain (production split,
  §4) and a conformal-coverage re-check before any production change.

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
