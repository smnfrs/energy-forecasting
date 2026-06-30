# Roadmap

A backlog of model and infrastructure improvements to revisit after the first pass at the merged repo is complete. Each entry sketches the problem, a proposed direction, expected impact, and open questions — not a committed plan. Sections will grow as later modelling stages produce their own follow-ups.

---

## Generation and Load Models

### Capacity-factor target for wind and solar

**Problem.** The current target is raw MWh per region per hour. Installed capacity has grown materially over the training window (onshore wind in Germany has added several GW in the last few years; solar has grown faster still). The same weather conditions therefore produce different absolute output across years, and the model has to implicitly learn a capacity time-trend on top of the weather→output relationship. Trends are hard to extrapolate cleanly past the training window — the symptom is a systematic negative bias on recent high-output hours, where the model's implicit ceiling lags reality.

**Proposal.** Model `capacity_factor = output_MWh / installed_capacity_MW` instead of raw output. At inference, predict cf and rescale by current installed capacity per region.

**Why this might help — including the data angle:**
- The target becomes stationary across capacity expansion, so the weather→output relationship is what the model actually learns. Calibration improves at high output, where the gap is currently largest.
- Per-region capacity normalisation makes a "wind day at 0.45 cf" comparable across TSOs in a way that raw MWh is not. This unlocks **cross-region pooling**: the four onshore TSOs become one combined training set, which is roughly 4× the rows for any given FE configuration. Same for solar across its four regions. With more data per fitted model, we should be able to support the larger search spaces (and trial counts) that the linear models in particular benefit from.
- Bounded target in roughly [0, 0.5] for wind onshore, [0, 0.6] for offshore, [0, 0.3] for solar — likely beneficial for ElasticNet specifically.
- Optionally opens older training data. If cf-as-target proves to generalise across regions, it should also generalise across time windows where we have actuals but no `hist_forecast` weather (pre-2022). Including that data means accepting the train-test weather-source mismatch (training on archived actual weather, predicting on issued forecasts), which we know hurts — so this is a downstream optional, not part of the first cut.

**Implementation sketch.**
1. Add a per-TSO installed-capacity time series (monthly granularity is sufficient — capacity changes slowly). Source candidates: ENTSO-E Transparency Platform (installed-capacity-per-production-type reports per bidding zone / TSO) and BNetzA Marktstammdatenregister aggregates. This is independent of `eu_locations.json`, which is for spatial weather aggregation only.
2. Compute `target_cf` at the data-loading step: `df["target_cf"] = df["target_mwh"] / capacity_at(timestamp, region)`.
3. Reuse the existing target-transform / target-lag / ensembling / MAPIE pipeline — they all work unchanged on cf.
4. At forecast time, scale predictions by `capacity_at(forecast_timestamp, region)`.
5. Pool TSOs into a single model with a region indicator. Keep per-TSO models as fallback during transition / for ablation.

**Open questions.**
- Granularity: per-TSO monthly is the obvious starting point. Per-bidding-zone (DE-LU national) only might be enough if the model relies on the regional split for spatial structure rather than absolute scale.
- Pooling vs separate models per TSO: pooling is the right default once cf normalises the targets, but worth keeping per-TSO as ablation in case region-specific structure (terrain, wind regime) carries enough signal that pooling washes out.
- This is a deviation from upstream EMA, which models raw MWh per TSO. EMA's accuracy is already exceeded on 4/5 targets, so this is an improvement on top of a strong baseline, not a fix for a broken design.

### Wind farm location metadata — sample representativeness

**Problem.** `data/locations/eu_locations.json` lists 14 onshore wind farms across all of Germany. The file is a hand-curated geographic sample used as **weather-query points** for spatial feature aggregation: each location becomes a weather-data query, then per-location weather features are collapsed into a per-region feature using weights (`capacity`, `n_turbines`, `idw`, etc.). The weights are relative, so the sample size and absolute capacity don't matter for spatial aggregation per se.

What's worth checking:
- Sample representativeness, especially in DE_TENNET (largest TSO, drives most of the wind onshore residual error). If the sample over-represents older farms or particular sub-regions, recent expansion is not reflected in spatial weights, and the model may over-weight legacy locations.
- The file has no provenance metadata — sample date / source registry are not recorded. Worth annotating once we touch it.

**Proposal.** Spot-check sample coverage against Marktstammdatenregister (which has asset-level data) before a full refresh — this may turn out to be a non-issue, in which case we leave it alone. If coverage is materially skewed, expand the sample or replace it with a clustered subsample of the registry (clustering keeps the weather-data fetch tractable). Independent of the capacity-factor track above, which uses TSO-level aggregates.

### Search-space refinement

**Problem.** Boolean and categorical Optuna search spaces for the weather FE classes contain options that, in practice, are never chosen by winning trials. Across the most recent training pass (48 base-model configs across the gen/load targets), several spatial-aggregation options have a 0% selection rate:

| FE class | Never-selected options |
|---|---|
| wind | `spatial_agg_method`: `distance_capacity`, `distance_n_turbines` |
| solar | `spatial_agg_method`: `mean`, `max`, `distance_capacity` |
| load | `spatial_agg_method`: `idw` |

These options consume search budget and add near-equivalent siblings that can confuse TPE, without ever winning. A handful of other options were chosen exactly once across 12–18 trials (`wind.spatial_agg_method`: `mean`, `max`, `idw`, `n_turbines`; `solar.cloud_lags_option` / `shortwave_lags_option`: `medium`) — borderline cases that are plausibly TPE noise rather than genuine optima.

**Proposal.**
- Remove the six options with zero selections. Trivial code change.
- Optional, more aggressive: collapse `wind.spatial_agg_method` to `[None, capacity]` (the two by far most often selected) and drop `medium` from the solar lag options.
- Narrow `load.cdh_threshold` from 20–26°C to 20–23°C — winning selections cluster at the low end.

**Expected impact.** Minor. The same configurations remain reachable, TPE convergence is marginally faster. Mostly hygiene; won't move headline metrics.

### Trial-budget allocation by FE-class dimensionality

**Problem.** Trial count is currently a flat 70 per (target, region, model_type) regardless of the FE class's search-space size. The wind FE class has 14 dimensions; solar 17; load 24. At 70 trials TPE gets 4–5 evaluations per dimension for solar and ~3 for load, which is at the lower end of where TPE consistently beats random sampling.

The empirical signature is that the more recently expanded search spaces (solar, load) showed a small mean regression in cv_mae vs prior narrower-search runs, while the wind class improved — consistent with TPE under-sampling in higher-dimensional spaces.

**Proposal.** Scale trials with dimensionality:
- wind (14 dims): 70 (no change)
- solar (17 dims): 120
- load (24 dims): 120

Total cost on a tower with 4-way parallel training is roughly an extra 5 hours — still a single overnight run.

**Open questions.**
- Optuna pruning would let us be smarter about budget allocation but the per-trial wall-clock varies modestly across the search space, so the value is bounded.
- If cross-region pooling lands (capacity-factor track), the count of distinct optimisations drops by ~4×, which mostly absorbs the higher per-optimisation budget.

### Wind onshore — diagnose the residual gap

**Problem.** Wind onshore is the one DE_NATIONAL target where we still trail EMA on the apples-to-apples backtest (~14% MAE gap). The error is concentrated, not diffuse:
- Top wind-output quintile: ~30% worse MAE vs EMA.
- Winter (Dec–Feb): ratio 1.2–1.3. Summer: ratio ≈ 1.0.
- Night hours (21–04): ratio 1.2–1.3. Midday (11–15): ratio 1.0.
- Bias signature: EMA over-forecasts by ~+270 MWh on average; ours under-forecasts by ~−250 MWh — coherent under-prediction across the three large TSOs (DE_50HZ, DE_AMPRION, DE_TENNET), with DE_TENNET driving most of the national-error variance.

The signature (under-prediction at high output, in winter, at night) is most consistent with the model's implicit output ceiling lagging actual installed capacity — i.e. the same problem that motivates the capacity-factor target above. Other plausible contributing factors: per-region geographic coverage of the location sample (especially DE_TENNET), and any systematic weather-source bias for north-Germany regions.

**Proposal.** Defer dedicated investigation until after capacity-factor and metadata refresh have landed, since both address the leading hypothesis. If the gap persists, the next dive is per-farm coverage in DE_TENNET and weather-source residuals for north-Germany hours.

**Open questions.**
- Does EMA use exog features or training-data preprocessing that we don't (or vice versa)? Worth a structural diff once the easier hypotheses are tested.

---

## Price Models

### Test an L2 objective across all tree families

**Problem.** Every price tree currently trains on an L1/MAE objective (LightGBM `objective="mae"`, XGBoost `reg:absoluteerror`, CatBoost `loss_function="MAE"`), inherited from EP for cross-repo comparability. The 2026-06-06 LGBM root-cause work found that L1 is the *weaker* objective for LightGBM specifically — L1 has a zero hessian, which degrades LightGBM's histogram split-finding and Newton-style leaf updates — and switching that one model to L2 (`objective="regression"`) gave the best cv_mae of any tree config on `price_fs_rfecv_optimum` (cv 20.22 vs 21.74 for the L1 probe). We kept MAE for now so the merged repo stays comparable to EP, but the gain suggests L2 (or Huber/quantile) may be underexplored across the whole tree set.

**Proposal.** Once EP-comparability is no longer a hard constraint, sweep the loss objective as a first-class search axis for each tree family — at minimum L2 vs L1, and ideally Huber with a tuned delta — rather than pinning it. Likely most impactful for LightGBM; XGBoost's `reg:absoluteerror` is purpose-built for L1 and may not benefit, and CatBoost's L1 was competitive in the real run.

**Expected impact.** Moderate for LightGBM (the 2026-06-06 evidence is ~1.5 cv-MAE on one feature set, smaller on holdout); uncertain but probably small for XGB/CatBoost. Worth a controlled sweep before committing.

**Open questions.**
- Does the L2 advantage survive sample-weighting and the full feature-selected ensemble, or is it a single-model artifact?
- If objectives diverge by family, does that *help* ensemble diversity (different loss → different error structure) or just shuffle which model wins?

---
