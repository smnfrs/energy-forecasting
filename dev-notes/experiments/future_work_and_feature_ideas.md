# Future work & feature ideas (price forecasting)

A running backlog of ideas worth pursuing but out of scope for the current
experiment. Not a plan — a place to park thoughts so they are not lost. Promote
an item to its own `dev-notes/experiments/*.md` plan when it gets picked up.

---

## Features to capture evening spikes / volatility

Context: the evening peak (≈17–20 UTC) is both the worst-forecast and the
highest-variance block, and its error is ~96% variance, not bias (see
[`loss_function_and_bias_study.md`](loss_function_and_bias_study.md) §1d). Loss
changes fix calibration; **reducing the evening error itself needs features that
explain scarcity**. Current features are largely load/wind/solar/calendar; the
scarcity drivers below are mostly absent. Roughly in order of expected value:

1. **Generator outages / availability (REMIT / ENTSO-E transparency).** Unplanned
   and planned conventional outages tighten supply and are the classic trigger for
   evening scarcity spikes. Almost certainly the single biggest missing lever.
   `eu_locations.json` is only a 1.16 GW sample (see memory), so we have no real
   availability signal today.
2. **Reserve margin / supply tightness indicator.** Forecast residual load minus
   available conventional + import capacity → a nonlinear "how tight is the
   evening" signal. Spikes are a threshold phenomenon; a margin feature lets the
   model learn the threshold.
3. **Residual-load ramp features.** Magnitude and steepness of the evening net-load
   ramp (load minus wind+solar) over the 2–4 h into the peak, the solar drop-off
   gradient, and the peak-residual-load hour. Ramps stress flexible capacity and
   drive the price gradient.
4. **Cross-border coupling.** Interconnector available capacity, neighbouring-zone
   prices / net position. Imports cap the German price; scarcity bites hardest when
   neighbours are simultaneously tight.
5. **Marginal-fuel cost & spreads.** Gas (TTF), coal, CO₂ (EUA) → clean-spark /
   clean-dark spread. These set the *level* of the peak price; interact them with
   the tightness signal (fuel cost × scarcity).
6. **Dunkelflaute / weather-extreme indicators.** Low-wind-and-low-solar flags,
   temperature (heating/cooling demand), and their multi-hour persistence — the
   textbook driver of evening scarcity.
7. **Storage / flexibility state.** Hydro reservoir levels, pumped-hydro and
   battery availability — flexibility that damps spikes when present.
8. **Volatility / regime features.** Recent realised price volatility, lagged
   spike indicators, day-type (holiday/bridge day), season — let the model widen
   its own uncertainty in spike-prone conditions.
9. **Explicit peak interactions.** hour × residual-load, hour × ramp, weekday ×
   evening — cheap, and directly targets the block where the bias/variance lives.

### Prerequisite: ENTSO-E Transparency ingestion (not yet in the repo)

Most of the high-value items above (1 outages, 2 reserve margin, 4 cross-border)
live on the **ENTSO-E Transparency Platform**, along with generation-by-type. As
of 2026-07-24 there is **no ENTSO-E data loader in the repo** — confirmed by grep;
the only `entso`/`outage` hits are unrelated. Simon **has an API key**, but the
ingestion path is unbuilt.

So the top-ranked evening-spike features are **blocked on building an ENTSO-E
loader first** — treat that as its own backlog item (own `dev-notes/experiments/`
plan when picked up):
- outage/availability series (unplanned + planned), aggregated to a supply-tightness signal;
- cross-border scheduled/available capacities and neighbouring-zone data;
- generation-by-type (cross-check against the existing SMARD/TSO parquets).

Commodity feeds (TTF gas, EUA carbon) for item 5 are a separate source, also not
yet ingested. Building ENTSO-E ingestion unlocks 1, 2 and 4 together, so it is the
natural first move on the feature track.

## Model-structure ideas (beyond features & loss)

- **Two-stage spike model:** a classifier for "is this a spike hour" feeding a
  regressor, or a hurdle/mixture model for the upper tail.
- **Quantile / expectile head for the tail** rather than a single point forecast,
  with the dashboard point taken as a chosen expectile in peak hours.
- **Separate peak vs off-peak models** — different features and loss for the
  17–20 UTC block, which behaves qualitatively differently from the rest of the day.

## Other backlog items

- Revisit the ensemble **weighting metric** (inverse-MAE) more broadly than the
  loss study covers — see that study's Phase 2.
- Dashboard: surface the per-hour ME/bias on a longer window than the current
  ~5-day production history so the calibration picture is stable (the study used
  the 11-year OOF; the live chart is noisy on 5 days).
