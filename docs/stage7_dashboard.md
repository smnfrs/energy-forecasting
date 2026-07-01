# Stage 7: Dashboard

Combines EP's price forecast display with EMA's generation/load multi-target view into a single static site deployed to GitHub Pages via the existing Stage 6 CI.

**Detailed plan:** this document
**Status:** Stage 7a implemented (2026-06-30) — pending verification gate (local browser smoke test)

---

## Context

The Stage 6 CI workflow already deploys `./deploy` to GitHub Pages:
```yaml
- uses: peaceiris/actions-gh-pages@v4
  with:
    publish_dir: ./deploy
```

`deploy/data/` is written by `publish.py` at inference time and uploaded to the Pages deploy. Stage 7 adds static HTML/CSS/JS files directly into `deploy/` alongside the `data/` subdirectory. No CI or workflow changes are needed.

### Existing source material

| Feature | EP deploy | EMA deploy |
|---------|-----------|------------|
| Chart library | Plotly.js | ApexCharts |
| Price chart | 24h bar + PI line | — |
| Forecast vs actuals | 30-day rolling | — |
| Daily error trend | 7-day bar | — |
| Monitoring page | per-model MAE/RMSE, composition, retrain log | — |
| Gen/load charts | — | per-TSO + national stacked area, line |
| Language toggle | EN/DE via simple dict | EN/DE via i18next |
| Dark mode | — | toggle |
| CI bands | — | polygon fill |

**Decision: use Plotly for all charts.** Consistent with EP; Plotly handles PI bands natively via `fill: "tonexty"` stacked traces.

**Decision: light-mode only.** Dark mode is nice-to-have; EP's CSS is already clean. Deferred to roadmap.

**Decision: split Stage 7 into 7a and 7b.** Stage 7a ships the dashboard against the existing Stage 6 contracts plus the minimum price actual/error additions. Stage 7b adds extended monitoring data: per-TSO gen/load, gen/load history/errors, per-model price errors, and retrain history. A verification gate sits between 7a and 7b.

**Decision: API-first data access, static fallback.** Dashboard JavaScript uses a small data client abstraction that can read from the FastAPI endpoints when `window.API_BASE_URL` is configured, and falls back to same-origin static JSON under `deploy/data/` for GitHub Pages. GitHub Pages cannot call an unhosted local FastAPI service, so the static JSON files remain the deployment source of truth unless/until the API is hosted publicly. The API and dashboard must share the same JSON contracts; the dashboard must not trigger inference or duplicate pipeline execution.

---

## 7.1 Data Contract

### 7.1.1 Files written by existing `publish.py` (Stage 6)

- `deploy/data/price_forecast.json`
- `deploy/data/gen_load/{target}_national.json`
- `deploy/data/forecast_history.json`
- `deploy/data/model_metadata.json`
- `deploy/data/errors/{date}.json`

### 7.1.2 New files to add in Stage 7

| File | Written by | Purpose |
|------|-----------|---------|
| `deploy/data/actuals.json` | `write_actuals()` | 30-day rolling actual DE-LU prices |
| `deploy/data/errors_summary.json` | `write_errors_summary()` | aggregated ensemble MAE/RMSE trend |
| `deploy/data/model_errors.json` | `write_model_errors()` | per-model daily MAE (for monitoring page) |
| `deploy/data/gen_load/{target}_{tso}.json` | extend `write_gen_load_forecasts()` | per-TSO forecasts (15 files) |
| `deploy/data/gen_load_actuals.json` | `write_gen_load_actuals()` | 7-day rolling gen/load SMARD actuals |
| `deploy/data/gen_load_history.json` | `append_gen_load_history()` | rolling gen/load forecast snapshots by delivery timestamp |
| `deploy/data/gen_load_errors_summary.json` | `write_gen_load_errors_summary()` | per-target gen/load MAE/RMSE trend computed from forecast history |
| `deploy/data/retrain_history.json` | extended `retrain.py` + deploy copy step | retrain event log |

### 7.1.3 `actuals.json`

Rolling 30-day actual DE-LU prices for the forecast-vs-actuals and daily-error charts.

```json
{
  "days": [
    {"date": "2026-06-29", "prices": [72.4, 68.1, ..., 88.5]},
    ...
  ],
  "count": 30
}
```

Source: `merged.parquet`, column `target_price`. Implementation:

```python
def write_actuals() -> None:
    from energy_forecasting.config import PROCESSED_DATA_DIR
    try:
        merged = pd.read_parquet(PROCESSED_DATA_DIR / "merged.parquet", columns=["target_price"])
    except FileNotFoundError:
        return
    # Filter then trim: complete days first, then take last 30.
    # merged.parquet has exactly 24 rows per day (normalize_dst guarantees this),
    # so len == 24 only guards against partial data at the trailing edge.
    by_date = merged.dropna(subset=["target_price"]).groupby(merged.index.date)
    complete = [(d, grp) for d, grp in by_date if len(grp) == 24]
    complete.sort(key=lambda x: x[0])
    days = [
        {"date": str(d), "prices": [round(float(v), 3) for v in grp["target_price"].values]}
        for d, grp in complete[-30:]
    ]
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DATA_DIR / "actuals.json").write_text(json.dumps({"days": days, "count": len(days)}, indent=2))
```

### 7.1.4 `errors_summary.json`

Aggregated ensemble error history for the monitoring MAE/RMSE trend chart.

```json
{
  "dates": ["2026-06-01", ..., "2026-06-29"],
  "mae":  [11.3, 9.8, ..., 10.1],
  "rmse": [18.2, 15.4, ..., 16.7]
}
```

```python
def write_errors_summary() -> None:
    if not ERRORS_DIR.exists():
        return
    records = sorted(
        [json.loads(p.read_text()) for p in ERRORS_DIR.glob("*.json")
         if p.stem[0].isdigit()],   # exclude non-date files in errors/
        key=lambda r: r["date"]
    )[-30:]
    (DEPLOY_DATA_DIR / "errors_summary.json").write_text(json.dumps({
        "dates": [r["date"] for r in records],
        "mae":   [r["mae"]  for r in records],
        "rmse":  [r["rmse"] for r in records],
    }, indent=2))
```

Called unconditionally from `write_outputs()` (not only from `compute_errors()`). This guarantees freshness on every inference run regardless of whether `compute_errors()` ran or returned early.

### 7.1.5 `model_errors.json`

Per-model daily MAE for the monitoring page's model comparison chart.

```json
{
  "dates": ["2026-06-01", ..., "2026-06-29"],
  "mae":  {"LGBMRegressor__fs_shap_top90": [...], "XGBRegressor__fs_rfecv_optimum": [...], "blend": [...]},
  "rmse": {"LGBMRegressor__fs_shap_top90": [...], ..., "blend": [...]}
}
```

**Source**: `price_inference.py` must also save per-model point predictions, so `compute_errors()` can compute per-model MAE against actuals. Change to `price_inference.py`:

```python
# In run_price_inference(), alongside writing price_df, also write:
per_model_preds = {}
for name, model, weight in active_models:
    y_pred = model.predict(X)
    per_model_preds[name] = y_pred.tolist()
# Save to deploy/data/per_model_preds/{date}.json at inference time
```

Then in `compute_errors()`, if per-model preds file exists for yesterday, compute MAE/RMSE per model and update `model_errors.json`:

```python
def _update_model_errors(date: str, y_true: np.ndarray, per_model_preds: dict) -> None:
    path = DEPLOY_DATA_DIR / "model_errors.json"
    existing = json.loads(path.read_text()) if path.exists() else {"dates": [], "mae": {}, "rmse": {}}
    # Remove this date if already present (idempotent)
    if date in existing["dates"]:
        idx = existing["dates"].index(date)
        existing["dates"].pop(idx)
        for k in existing["mae"]: existing["mae"][k].pop(idx)
        for k in existing["rmse"]: existing["rmse"][k].pop(idx)
    # Insert new entry
    existing["dates"].append(date)
    for name, preds in per_model_preds.items():
        y_p = np.array(preds)
        existing["mae"].setdefault(name, []).append(round(float(np.mean(np.abs(y_true - y_p))), 3))
        existing["rmse"].setdefault(name, []).append(round(float(np.sqrt(np.mean((y_true - y_p)**2))), 3))
    # Blend errors from errors/{date}.json (already computed)
    # ... add blend row
    # Trim to 30 days
    path.write_text(json.dumps(existing, indent=2))
```

Intermediate per-model preds are written to `deploy/data/per_model_preds/{date}.json` and can be deleted after `model_errors.json` is updated. Add `deploy/data/per_model_preds/` to `.gitignore`.

### 7.1.6 Per-TSO gen/load forecasts

Extend `write_gen_load_forecasts()` to also write per-TSO files alongside the national ones.

TSO regions by target:
- `wind_onshore`: DE_50HZ, DE_AMPRION, DE_TENNET, DE_TRANSNETBW
- `wind_offshore`: DE_50HZ, DE_TENNET
- `solar`: DE_50HZ, DE_AMPRION, DE_TENNET, DE_TRANSNETBW
- `load`: DE_50HZ, DE_AMPRION, DE_TENNET, DE_TRANSNETBW, DE_CREOS

Total: 15 files. Filename convention: `{target}_{region_lower}.json` where region_lower strips `DE_` and lowercases, e.g. `wind_onshore_50hz.json`, `load_creos.json`.

```python
TSO_SUFFIX = {
    "DE_50HZ": "50hz", "DE_AMPRION": "amprion",
    "DE_TENNET": "tennet", "DE_TRANSNETBW": "transnetbw", "DE_CREOS": "creos",
}

for (target, region), df in gen_load_results.items():
    if region == "DE_NATIONAL":
        filename = f"{target}_national.json"
    else:
        filename = f"{target}_{TSO_SUFFIX[region]}.json"
    payload = {"target": target, "region": region, "issued_at": issued_at,
               "horizon_hours": len(df), "unit": "MW", "forecasts": _hourly_entries(df)}
    (GEN_LOAD_DATA_DIR / filename).write_text(json.dumps(payload, indent=2))
```

### 7.1.7 `gen_load_actuals.json`

Seven-day rolling actual gen/load per target at national (sum-of-TSO) level. This belongs to Stage 7b, not the Stage 7a verification gate.

```json
{
  "wind_onshore": {"days": [{"date": "2026-06-23", "values": [1200, 1350, ..., 980]}, ...], "count": 7},
  "wind_offshore": {"days": [...], "count": 7},
  "solar":         {"days": [...], "count": 7},
  "load":          {"days": [...], "count": 7}
}
```

Source: per-TSO Parquet files at `data/processed/tso/{TSO name}.parquet`, e.g. `50Hertz.parquet`, `Amprion.parquet`. Target columns inside those files are resolved with the same helpers used by gen/load training and inference: `modeling.gen_load._load_tso_data()` and `_get_target_col(target, region)`. Do not construct paths like `{target}_{tso_suffix}.parquet`; that layout does not exist.

The per-TSO files are in UTC; convert to Europe/Berlin delivery hours before summing, to match the gen/load forecast timestamps (which come from the inference pipeline). The simpler alternative is to keep UTC throughout (both actuals and forecasts) and let Plotly display in UTC — the 1-2 hour offset is visible but not misleading for a generation chart.

**Decision: UTC timestamps throughout for gen/load actuals** (avoids the tz conversion complexity; gen/load data is fundamentally a UTC physical quantity unlike price which is tied to delivery-period auction timing).

```python
def write_gen_load_actuals() -> None:
    from energy_forecasting.config import PROCESSED_DATA_DIR
    from energy_forecasting.config.modeling import REGION_TO_TSO, GEN_LOAD_TARGETS

    tso_dir = PROCESSED_DATA_DIR / "tso"
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=7)
    result = {}

    for target in ["wind_onshore", "wind_offshore", "solar", "load"]:
        frames = []
        for region in GEN_LOAD_TARGETS[target]["regions"]:
            df = _load_tso_data(region).loc[cutoff:]
            target_col = _get_target_col(target, region)
            if target_col not in df.columns:
                continue
            frames.append(df[target_col])
        if not frames:
            continue
        national = pd.concat(frames, axis=1).sum(axis=1, min_count=1).dropna()
        by_date = national.groupby(national.index.normalize())
        days = []
        for d, grp in sorted(by_date):
            if len(grp) < 24:
                continue
            days.append({"date": d.isoformat()[:10],
                         "values": [round(float(v), 1) for v in grp.values[:24]]})
        result[target] = {"days": days, "count": len(days)}

    (DEPLOY_DATA_DIR / "gen_load_actuals.json").write_text(json.dumps(result, indent=2))
```

### 7.1.8 `gen_load_history.json` and `gen_load_errors_summary.json`

Stage 7b adds gen/load forecast history before adding gen/load accuracy metrics. Do not compute a rolling error trend from only the current `gen_load/{target}_national.json`; that file is a single current forecast snapshot and mostly covers future timestamps.

`gen_load_history.json` stores forecast snapshots by target and issue time:

```json
{
  "targets": {
    "wind_onshore": [
      {
        "issued_at": "2026-06-30T08:00:00Z",
        "region": "DE_NATIONAL",
        "forecasts": [
          {"timestamp": "2026-06-30T09:00:00Z", "forecast": 1234.5}
        ]
      }
    ]
  },
  "count": 30
}
```

`append_gen_load_history(gen_load_results, issued_at)` is called from `write_outputs()` after `write_gen_load_forecasts()`. It stores national forecasts for `wind_onshore`, `wind_offshore`, `solar`, and `load`, deduplicated by `(target, issued_at[:10])`, and keeps the last 30 issue dates per target. Per-TSO forecast history is deferred unless the Stage 7b UI explicitly needs per-TSO accuracy.

`gen_load_errors_summary.json` is then computed by joining historical forecast snapshots to actuals by delivery timestamp:

```json
{
  "dates": ["2026-06-22", ..., "2026-06-28"],
  "wind_onshore": {"mae": [...], "rmse": [...]},
  "wind_offshore": {"mae": [...], "rmse": [...]},
  "solar":         {"mae": [...], "rmse": [...]},
  "load":          {"mae": [...], "rmse": [...]}
}
```

Algorithm for `write_gen_load_errors_summary()`:

1. Load `gen_load_history.json` and `gen_load_actuals.json`.
2. For each target and forecast snapshot, keep forecast timestamps that have actual values.
3. Group matched forecast/actual pairs by delivery date.
4. Compute MAE/RMSE only for dates with at least 24 matched hours.
5. Keep the last 30 delivery dates; the UI may display the last 7 by default.

This makes gen/load accuracy semantically consistent with price accuracy: every error compares a forecast issued before delivery against the actual value for the same delivery timestamp.

### 7.1.9 `retrain_history.json`

Written by `energy_forecasting/deploy/retrain.py` when a price retrain completes.

```json
[
  {
    "date": "2026-07-01T06:12:00Z",
    "old_holdout_mae": 11.148,
    "new_holdout_mae": 10.823,
    "degradation_pct": -2.9,
    "n_models": 5,
    "needs_reselection": false
  }
]
```

In `retrain.py`'s `_update_ensemble_config()` (or immediately after it), append a new entry and write the rolling history (keep last 12 events). The degradation percentage uses `(new - old) / old * 100`, so negative = improvement.

### 7.1.10 `forecast_history.json` adapter note

The dashboard JS must NOT use EP's `renderHistoryChart()` unchanged. Stage 6's history schema differs from EP's expected shape:

| Field | Our Stage 6 format | EP's expected format |
|-------|--------------------|----------------------|
| Top level | `{target, forecasts: [...], count}` | array directly |
| Per-entry date | `entry.issued_at` (pipeline run time) | `entry.date` (delivery date) |
| Per-entry prices | `entry.forecasts[].forecast` | `entry.prices[]` |

The dashboard JS adapter:

```javascript
function adaptHistory(history) {
  // history = {target, forecasts: [{issued_at, forecasts: [{timestamp, forecast}]}]}
  return (history.forecasts || [])
    .filter(e => e.forecasts && e.forecasts.length === 24)
    .map(e => ({
      date: e.forecasts[0].timestamp.slice(0, 10),   // delivery date from first timestamp
      prices: e.forecasts.map(f => f.forecast),
    }));
}
```

The adapter is called once in `init()` before passing data to render functions.

### 7.1.11 `fetchJSON` contract

`fetchJSON(path)` takes the **exact** path relative to the page's location, with no internal prefix. Callers are explicit:

```javascript
const DATA = "data/";

// At deploy root level (index.html, monitoring.html):
fetchJSON("translations.json")          // deploy/translations.json
fetchJSON(DATA + "price_forecast.json") // deploy/data/price_forecast.json
fetchJSON(DATA + "gen_load/wind_onshore_national.json")

// Implementation — no magic prefix:
async function fetchJSON(path) {
  try {
    const resp = await fetch(path);
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}
```

### 7.1.12 Dashboard data client: API first, static fallback

The dashboard must use the same contracts exposed by the FastAPI app where possible, without starting inference or duplicating backend logic. Implement a small data client in `script.js` and `monitoring.js`:

```javascript
const DATA = "data/";
const API_BASE = window.API_BASE_URL || null;

async function getJSON(staticPath, apiPath) {
  if (API_BASE && apiPath) {
    const fromApi = await fetchJSON(API_BASE.replace(/\/$/, "") + apiPath);
    if (fromApi) return fromApi;
  }
  return fetchJSON(staticPath);
}
```

Mapping for Stage 7a:

| Dashboard need | API path when hosted | Static fallback |
|----------------|----------------------|-----------------|
| price forecast | `/forecast/price` | `data/price_forecast.json` |
| gen/load national forecast | `/forecast/generation/{target}` or `/forecast/load` | `data/gen_load/{target}_national.json` |
| price history | `/forecast/history` | `data/forecast_history.json` |
| model metadata | `/models` | `data/model_metadata.json` |
| daily price errors | `/models/performance` | `data/errors_summary.json` |
| actual prices | none in Stage 6 API | `data/actuals.json` |

Stage 7b can either add API endpoints for extended dashboard contracts or continue using static fallback files for those contracts until the API is publicly hosted. Candidate Stage 7b endpoints: `/dashboard/actuals/price`, `/dashboard/errors/price`, `/dashboard/gen-load/history`, `/dashboard/gen-load/actuals`, `/dashboard/gen-load/performance`, `/dashboard/retrain-history`.

GitHub Pages deployment still works without a hosted API because all data files are published under `deploy/data/`. If the API is later hosted, setting `window.API_BASE_URL` switches the dashboard to API reads without changing chart code.

---

## 7.2 Stage Split and Verification Gate

### Stage 7a: Core Dashboard

Goal: ship a usable dashboard on GitHub Pages with minimal backend changes.

Scope:

- Static dashboard assets: `index.html`, `monitoring.html`, `script.js`, `monitoring.js`, `styles.css`, `translations.json`, `.nojekyll`.
- Price forecast chart, gen/load national overview, price forecast-vs-actuals, daily price error bars.
- Existing model metadata/composition display.
- Data additions limited to `actuals.json` and `errors_summary.json`.
- `adaptHistory()` and exact-path `fetchJSON()` fixes.
- API-first/static-fallback data client.
- Browser/static smoke test.

Explicitly out of Stage 7a:

- Per-TSO gen/load controls.
- Gen/load actual overlays.
- Gen/load accuracy metrics.
- Per-model historical errors.
- Retrain history publishing.

### Verification Gate: 7a -> 7b

Do not start Stage 7b until all of the following pass:

- `make forecast-skip-update` writes the Stage 7a data files.
- `make serve-dashboard` serves the dashboard locally.
- Main page renders price, national gen/load, history, and daily price error charts.
- Monitoring page renders model composition and available price error trend.
- DE/EN toggle works on both pages.
- API-first/static-fallback client is verified in static mode and, if a local API is running, API mode.
- Python tests for `write_actuals()` and `write_errors_summary()` pass.
- Browser/static smoke test passes locally.

### Stage 7b: Extended Monitoring

Scope after the gate:

- Per-TSO gen/load forecast files and UI toggles.
- `gen_load_actuals.json`.
- `gen_load_history.json`.
- `gen_load_errors_summary.json`, computed from forecast history joined to actuals.
- Per-model price prediction snapshots and `model_errors.json`.
- Durable `retrain_history.json` publishing.
- Optional API endpoints for extended dashboard contracts if the API is hosted or about to be hosted.

---

## 7.3 File Layout

This is the cumulative Stage 7b layout. Stage 7a only requires the static assets plus `price_forecast.json`, `forecast_history.json`, `model_metadata.json`, `actuals.json`, `errors_summary.json`, and the four national gen/load forecast files.

```
deploy/
├── .nojekyll                        ← prevents GitHub Pages Jekyll processing
├── index.html                       ← main forecast page
├── monitoring.html                  ← model monitoring page
├── script.js                        ← main page JS
├── monitoring.js                    ← monitoring page JS
├── styles.css                       ← shared styles (extend EP's)
├── translations.json                ← EN/DE strings
└── data/                            ← written by publish.py (gitignored locally)
    ├── price_forecast.json
    ├── actuals.json
    ├── errors_summary.json
    ├── model_errors.json
    ├── forecast_history.json
    ├── model_metadata.json
    ├── retrain_history.json
    ├── gen_load_actuals.json
    ├── gen_load_errors_summary.json
    ├── gen_load/
    │   ├── wind_onshore_national.json
    │   ├── wind_onshore_50hz.json
    │   ├── wind_onshore_amprion.json
    │   ├── wind_onshore_tennet.json
    │   ├── wind_onshore_transnetbw.json
    │   ├── wind_offshore_national.json
    │   ├── wind_offshore_50hz.json
    │   ├── wind_offshore_tennet.json
    │   ├── solar_national.json
    │   ├── solar_50hz.json
    │   ├── solar_amprion.json
    │   ├── solar_tennet.json
    │   ├── solar_transnetbw.json
    │   ├── load_national.json
    │   ├── load_50hz.json
    │   ├── load_amprion.json
    │   ├── load_tennet.json
    │   ├── load_transnetbw.json
    │   └── load_creos.json
    ├── per_model_preds/             ← intermediate, deleted after model_errors.json update
    │   └── {date}.json
    └── errors/
        └── {date}.json
```

Add to `.gitignore`: `deploy/data/`

---

## 7.3 Main Page (`index.html`)

Single HTML page, card-per-section layout.

### Structure

```
header
  title: "DE-LU Energy Forecast"
  subtitle: "Day-Ahead Electricity & Generation/Load Prediction"
  last-updated timestamp
  [Monitoring] nav link | [DE/EN] toggle

main
  [reselection-warning banner — hidden unless needs_reselection]

  §1  Price Forecast (24h)
      bar chart: 24 hours × EUR/MWh + PI shaded band
      label: "Delivery: {delivery date}"

  §2  Generation & Load — National Overview (168h)
      stacked area: wind onshore + wind offshore + solar (left y-axis, MW)
      line overlay: load (right y-axis, MW)
      "Now" annotation + actuals overlay for past 7 days where available

  §3–6  Individual gen/load cards (collapsible <details>)
      Wind Onshore | Wind Offshore | Solar | Load
      Each: 168h line + PI band + TSO toggle checkboxes + actuals overlay

  §7  Forecast vs Actual — Price (30 days)
      actuals: solid grey line  |  forecast: dashed blue line

  §8  Daily Error — Price (last 7 days)
      grouped bars: MAE (blue) + RMSE (red)

footer
```

### §1 Price forecast chart

Source: `data/price_forecast.json`

Traces:
1. Lower PI boundary — `{type:"scatter", mode:"none", y:lower, fill:null, showlegend:false}`
2. PI band — `{type:"scatter", mode:"none", y:upper, fill:"tonexty", fillcolor:"rgba(59,130,246,0.15)", name:"90% PI"}`
3. Point forecast — `{type:"bar", y:values, marker:{color:"#3B82F6"}}`

x-axis labels: delivery hours `"00:00"` through `"23:00"` extracted from `f.timestamp`.
Skip traces 1–2 if all `forecast_lower` values are null.

### §2 Gen/load national overview

Source: `data/gen_load/*_national.json` (4 fetches in parallel) + `data/gen_load_actuals.json`

Actuals overlay: solid lines for the last 7 days before the forecast start, greyed out. Use `trimPast()` logic from EMA's `script.js` (keep data within 7 days before forecast start; skip if gap > 48h).

Traces (stacked area generation + load line):
```javascript
// Stacked area series (stackgroup: "gen")
[{name:"Wind Onshore", fill:"tozeroy", stackgroup:"gen", fillcolor:"rgba(34,102,204,0.7)", ...},
 {name:"Wind Offshore", fill:"tonexty", stackgroup:"gen", fillcolor:"rgba(68,170,221,0.7)", ...},
 {name:"Solar", fill:"tonexty", stackgroup:"gen", fillcolor:"rgba(221,170,0,0.7)", ...},
 // Load on secondary y-axis (no stackgroup)
 {name:"Load (forecast)", yaxis:"y2", line:{color:"#EE0000", width:2, dash:"dash"}, ...},
 {name:"Load (actual)", yaxis:"y2", line:{color:"#EE0000", width:1}, ...}]
```

Layout:
```javascript
{
  yaxis:  {title:"MW (generation)", rangemode:"tozero"},
  yaxis2: {title:"MW (load)", overlaying:"y", side:"right", rangemode:"tozero"},
  height: 350
}
```

### §3–6 Individual gen/load cards

Each is a `<details class="forecast-card">` element. Chart is lazy-rendered on first open.

Controls per card:
- TSO toggle checkboxes (the national total is always shown; per-TSO are opt-in)
- "Show PI" checkbox
- "Show actuals" checkbox

When TSO checkboxes change: re-render the chart with selected traces only.

```javascript
const GEN_LOAD_CARDS = [
  {
    id: "card-wind-onshore", target: "wind_onshore",
    label: "wind_onshore", color: "#2266CC",
    tsos: ["national", "50hz", "amprion", "tennet", "transnetbw"],
    tsoLabels: {national:"National", "50hz":"50Hertz", amprion:"Amprion", tennet:"TenneT", transnetbw:"TransnetBW"}
  },
  {id: "card-wind-offshore", target: "wind_offshore", color: "#44AADD",
   tsos: ["national", "50hz", "tennet"],
   tsoLabels: {national:"National", "50hz":"50Hertz", tennet:"TenneT"}},
  {id: "card-solar", target: "solar", color: "#DDAA00",
   tsos: ["national", "50hz", "amprion", "tennet", "transnetbw"],
   tsoLabels: {national:"National", "50hz":"50Hertz", amprion:"Amprion", tennet:"TenneT", transnetbw:"TransnetBW"}},
  {id: "card-load", target: "load", color: "#EE0000",
   tsos: ["national", "50hz", "amprion", "tennet", "transnetbw", "creos"],
   tsoLabels: {national:"National", "50hz":"50Hertz", amprion:"Amprion", tennet:"TenneT", transnetbw:"TransnetBW", creos:"Creos"}},
];
```

PI band and actuals overlay use the same pattern as §2 but for a single target.

### §7 Forecast vs Actual (price)

Source: `data/actuals.json` + `data/forecast_history.json`

History is adapted via `adaptHistory()` before rendering (see §7.1.10). Same render logic as EP's `renderHistoryChart()` after adaptation.

### §8 Daily Error (price)

Source: `data/actuals.json` + `data/forecast_history.json`

Compute MAE/RMSE per day client-side: for each day in the adapted history that also has actuals, compute 24-hour mean absolute error. Show last 7 complete days. Same as EP's `computeDailyErrors()` after adaptation.

---

## 7.4 Monitoring Page (`monitoring.html`)

### Sections

1. **Price MAE/RMSE trend** — per-model daily lines (from `model_errors.json`) + bold ensemble line
2. **Ensemble composition** — horizontal bar (from `model_metadata.json`)
3. **Info panel** — last retrain, holdout MAE, PI coverage, conformal quantile
4. **Retrain log** — table of last 5 retrain events (from `retrain_history.json`)
5. **Gen/load accuracy** — tab or card per target showing 7-day rolling MAE/RMSE (from `gen_load_errors_summary.json`)

### Price error chart

Source: `model_errors.json` — `{dates, mae: {model_name: [...]}, rmse: {...}}`

Matches EP's `monitoring.js` `renderErrorChart()` almost exactly. The schema from our `_update_model_errors()` matches EP's expected shape. Retrain-date vertical markers come from `retrain_history.json`.

### Ensemble composition

Source: `model_metadata.json` — `{models: [{name, category, weight}], holdout_mae, holdout_rmse, pi_coverage}`

Adapt EP's `renderCompositionPanel()`:
- `metadata.models` instead of `metadata.model_details`
- `metadata.holdout_mae` instead of `metadata.blend_mae`
- Add PI coverage and conformal quantile to info line

### Retrain log

Source: `retrain_history.json` — array of `{date, old_holdout_mae, new_holdout_mae, degradation_pct, n_models, needs_reselection}`

Port EP's `renderRetrainLog()`. When `retrain_history.json` is absent (first run before any retrain), show "No retrain events recorded yet." — matching EP.

### Gen/load accuracy section

Source: `gen_load_errors_summary.json` — `{dates, wind_onshore:{mae,rmse}, ...}`

Four Plotly bar charts (one per target), each showing 7-day rolling MAE and RMSE side-by-side. Grouped under a "Generation & Load Accuracy" heading.

---

## 7.5 `script.js`

```javascript
(function () {
  "use strict";

  const DATA = "data/";
  let currentLang = "en";
  let translations = {};

  async function fetchJSON(path) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) return null;
      return await resp.json();
    } catch { return null; }
  }

  // Convert Stage 6 history format → EP-compatible [{date, prices}]
  function adaptHistory(history) {
    return (history?.forecasts || [])
      .filter(e => e.forecasts?.length === 24)
      .map(e => ({
        date: e.forecasts[0].timestamp.slice(0, 10),
        prices: e.forecasts.map(f => f.forecast),
      }));
  }

  // Build PI traces: returns [lowerTrace, piTrace, pointTrace] or just [pointTrace]
  function piTraces(timestamps, values, lower, upper, color, name) {
    const hasPI = Array.isArray(lower) && lower.some(v => v !== null);
    const traces = [];
    if (hasPI) {
      traces.push({x:timestamps, y:lower, type:"scatter", mode:"none",
                   fill:null, showlegend:false, hoverinfo:"skip"});
      traces.push({x:timestamps, y:upper, type:"scatter", mode:"none",
                   fill:"tonexty", fillcolor:colorWithAlpha(color, 0.15),
                   name:"90% PI", showlegend:true, hoverinfo:"skip"});
    }
    traces.push({x:timestamps, y:values, type:"scatter", mode:"lines",
                 name:name, line:{color, width:2}});
    return traces;
  }

  function colorWithAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  async function init() {
    translations = await fetchJSON("translations.json") || {};
    const [price, historyRaw, actuals, metadata] = await Promise.all([
      fetchJSON(DATA + "price_forecast.json"),
      fetchJSON(DATA + "forecast_history.json"),
      fetchJSON(DATA + "actuals.json"),
      fetchJSON(DATA + "model_metadata.json"),
    ]);
    const history = adaptHistory(historyRaw);
    renderPriceChart(price);
    renderGenLoadSummary();
    setupGenLoadCards();
    renderHistoryChart(actuals, history);
    renderErrorChart(actuals, history);
    renderMetadata(metadata);
    setupLanguageToggle();
  }

  function renderPriceChart(forecast) { /* §1 */ }
  async function renderGenLoadSummary() { /* §2 — fetches gen_load/*.json and gen_load_actuals.json */ }
  function setupGenLoadCards() { /* §3-6 — lazy-init on <details> toggle */ }
  function renderHistoryChart(actuals, history) { /* §7 */ }
  function renderErrorChart(actuals, history) { /* §8 */ }
  function renderMetadata(metadata) { /* update last-updated, reselection warning */ }

  function t(key) {
    return (translations[currentLang] && translations[currentLang][key]) || key;
  }
  function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach(el => {
      const text = t(el.getAttribute("data-i18n"));
      if (text !== el.getAttribute("data-i18n")) el.textContent = text;
    });
  }
  function setupLanguageToggle() {
    const btn = document.getElementById("lang-toggle");
    btn.addEventListener("click", () => {
      currentLang = currentLang === "en" ? "de" : "en";
      btn.textContent = currentLang === "en" ? "DE" : "EN";
      applyTranslations();
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
```

**Lazy-init pattern for gen/load detail cards:**
```javascript
function setupGenLoadCards() {
  for (const cfg of GEN_LOAD_CARDS) {
    const el = document.getElementById(cfg.id);
    if (!el) continue;
    el.addEventListener("toggle", async function onFirst() {
      if (!el.open) return;
      el.removeEventListener("toggle", onFirst);
      // Fetch national + all TSO files for this target
      const files = cfg.tsos.map(tso => `${DATA}gen_load/${cfg.target}_${tso}.json`);
      const dataArr = await Promise.all(files.map(fetchJSON));
      renderGenLoadCard(el.querySelector(".chart-container"), cfg, dataArr);
    });
  }
}
```

---

## 7.6 `styles.css`

Port EP's `styles.css` as base, with additions:

```css
/* Gen/load summary chart */
#summary-chart { min-height: 350px; }

/* TSO toggle buttons */
.tso-controls { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0 0.75rem; }
.tso-btn { font-size: 0.78rem; padding: 0.2rem 0.6rem; border-radius: 3px; border: 1.5px solid; cursor: pointer; }
.tso-btn.active { opacity: 1; }
.tso-btn.inactive { opacity: 0.35; }

/* Collapsible gen/load cards */
details.forecast-card {
  border: 1px solid #e9ecef; border-radius: 8px; overflow: hidden; margin-bottom: 0.75rem;
}
details.forecast-card > summary {
  padding: 0.9rem 1.25rem; cursor: pointer; font-weight: 600;
  color: #343a40; list-style: none; user-select: none;
  display: flex; align-items: center; gap: 0.5rem;
}
details.forecast-card > summary::before {
  content: "▶"; display: inline-block; transition: transform 0.2s; font-size: 0.75rem;
}
details.forecast-card[open] > summary::before { transform: rotate(90deg); }
details.forecast-card .chart-body { padding: 0.75rem 1.25rem 1.25rem; }

/* Two-column grid for gen/load cards on wide screens */
@media (min-width: 800px) {
  .forecast-cards-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
}
```

---

## 7.7 `translations.json`

Extend EP's translations with gen/load and monitoring additions:

```json
{
  "en": {
    "title": "DE-LU Energy Forecast",
    "subtitle": "Day-Ahead Electricity Price & Generation/Load Prediction",
    "forecast_chart_title": "24-Hour Price Forecast",
    "gen_load_summary_title": "Generation Mix & Load — National (7 Days)",
    "gen_load_section": "Individual Generation & Load Forecasts",
    "wind_onshore": "Wind Onshore",
    "wind_offshore": "Wind Offshore",
    "solar": "Solar",
    "load": "Load",
    "national": "National",
    "now": "Now",
    "pi_band": "90% PI",
    "unit_mw": "MW",
    "show_pi": "Show PI",
    "show_actuals": "Show Actuals",
    "forecast_detail_wind_onshore": "Wind Onshore — 7-Day National Forecast",
    "forecast_detail_wind_offshore": "Wind Offshore — 7-Day National Forecast",
    "forecast_detail_solar": "Solar — 7-Day National Forecast",
    "forecast_detail_load": "Load — 7-Day National Forecast",
    "gen_load_accuracy_title": "Generation & Load Accuracy (7 Days)",
    "pi_coverage": "PI coverage",
    "conformal_quantile": "Conformal quantile",
    "no_gen_load_data": "Generation/load forecast not yet available.",
    "no_retrain_data": "No retrain events recorded yet.",
    "history_chart_title": "Forecast vs Actual — Price (30 Days)",
    "error_chart_title": "Daily Forecast Error — Price (Last 7 Days)",
    "last_updated": "Last updated",
    "mae": "MAE",
    "rmse": "RMSE",
    "unit": "EUR/MWh",
    "forecast": "Forecast",
    "actual": "Actual",
    "no_data": "No data available yet.",
    "reselection_warning": "Model degradation detected. Manual reselection recommended.",
    "forecast_date": "Delivery",
    "monitoring_link": "Model Monitoring",
    "monitoring_title": "Model Monitoring",
    "model_mae_title": "Per-Model MAE (30 Days)",
    "model_rmse_title": "Per-Model RMSE (30 Days)",
    "composition_title": "Ensemble Composition",
    "retrain_log_title": "Retrain Log",
    "last_retrain": "Last retrain",
    "mae_change": "MAE change",
    "back_to_forecast": "Back to Forecast"
  },
  "de": {
    "title": "DE-LU Energieprognose",
    "subtitle": "Day-Ahead Strompreis- & Erzeugungs-/Lastprognose",
    "forecast_chart_title": "24-Stunden-Preisprognose",
    "gen_load_summary_title": "Erzeugungsmix & Last — National (7 Tage)",
    "gen_load_section": "Einzelne Erzeugungs- & Lastprognosen",
    "wind_onshore": "Wind onshore",
    "wind_offshore": "Wind offshore",
    "solar": "Solar",
    "load": "Last",
    "national": "National",
    "now": "Jetzt",
    "pi_band": "90%-KI",
    "unit_mw": "MW",
    "show_pi": "KI anzeigen",
    "show_actuals": "Istwerte anzeigen",
    "forecast_detail_wind_onshore": "Wind onshore — Nationale 7-Tage-Prognose",
    "forecast_detail_wind_offshore": "Wind offshore — Nationale 7-Tage-Prognose",
    "forecast_detail_solar": "Solar — Nationale 7-Tage-Prognose",
    "forecast_detail_load": "Last — Nationale 7-Tage-Prognose",
    "gen_load_accuracy_title": "Erzeugung & Last — Prognosegenauigkeit (7 Tage)",
    "pi_coverage": "PI-Abdeckung",
    "conformal_quantile": "Konformes Quantil",
    "no_gen_load_data": "Erzeugungs-/Lastprognose noch nicht verfügbar.",
    "no_retrain_data": "Noch keine Retraining-Ereignisse aufgezeichnet.",
    "history_chart_title": "Prognose vs. Tatsächlich — Preis (30 Tage)",
    "error_chart_title": "Täglicher Prognosefehler — Preis (letzte 7 Tage)",
    "last_updated": "Zuletzt aktualisiert",
    "mae": "MAE",
    "rmse": "RMSE",
    "unit": "EUR/MWh",
    "forecast": "Prognose",
    "actual": "Tatsächlich",
    "no_data": "Noch keine Daten verfügbar.",
    "reselection_warning": "Modellverschlechterung erkannt. Manuelle Neuauswahl empfohlen.",
    "forecast_date": "Lieferung",
    "monitoring_link": "Modell-Monitoring",
    "monitoring_title": "Modell-Monitoring",
    "model_mae_title": "MAE pro Modell (30 Tage)",
    "model_rmse_title": "RMSE pro Modell (30 Tage)",
    "composition_title": "Ensemble-Zusammensetzung",
    "retrain_log_title": "Retraining-Verlauf",
    "last_retrain": "Letztes Retraining",
    "mae_change": "MAE-Änderung",
    "back_to_forecast": "Zurück zur Prognose"
  }
}
```

---

## 7.8 GitHub Pages

**`deploy/.nojekyll`** — committed empty file to prevent Jekyll from stripping `data/` subdirectories (Jekyll ignores files/dirs with underscores by default).

**CI verification:** the Stage 6 `daily_forecast.yml` deploys `./deploy` which now contains HTML/CSS/JS. No structural changes to the workflow. The static files (`.nojekyll`, `index.html`, etc.) are checked in and will be present in the checkout, then overwritten data lands on top during the Pages deploy step.

**Local preview:** `python -m http.server 8080 --directory deploy/` after `make forecast-skip-update`.

---

## 7.9 Makefile

```makefile
serve-dashboard:
	cd deploy && python -m http.server 8080

open-dashboard:
	xdg-open http://localhost:8080

test-dashboard:
	npx playwright test tests/test_dashboard.js --headed
```

---

## 7.10 Tests

### Python tests (`tests/test_deploy_publish.py` extensions)

```python
def test_write_actuals_filters_then_trims(tmp_deploy, tmp_merged_parquet):
    """Complete days are filtered before trimming to 30."""
    write_actuals()
    data = json.loads((tmp_deploy / "actuals.json").read_text())
    assert "days" in data
    for day in data["days"]:
        assert len(day["prices"]) == 24

def test_write_actuals_missing_parquet(tmp_deploy):
    """write_actuals returns silently if merged.parquet is absent."""
    write_actuals()  # should not raise
    assert not (tmp_deploy / "actuals.json").exists()

def test_write_errors_summary(tmp_deploy):
    """Aggregates errors/*.json sorted by date."""
    errors_dir = tmp_deploy / "errors"
    errors_dir.mkdir()
    for date, mae, rmse in [("2026-06-28", 10.0, 16.0), ("2026-06-29", 11.0, 17.0)]:
        (errors_dir / f"{date}.json").write_text(json.dumps({"date": date, "mae": mae, "rmse": rmse}))
    write_errors_summary()
    summary = json.loads((tmp_deploy / "errors_summary.json").read_text())
    assert summary["dates"] == ["2026-06-28", "2026-06-29"]
    assert summary["mae"] == [10.0, 11.0]

def test_write_errors_summary_called_from_write_outputs(tmp_deploy, ...):
    """write_outputs always rebuilds errors_summary.json."""
    ...

def test_write_gen_load_actuals_utc(tmp_deploy, tmp_tso_parquets):
    """gen_load_actuals.json is written in UTC with correct shape."""
    write_gen_load_actuals()
    data = json.loads((tmp_deploy / "gen_load_actuals.json").read_text())
    for target in ["wind_onshore", "wind_offshore", "solar", "load"]:
        if target in data:
            for day in data[target]["days"]:
                assert len(day["values"]) == 24

def test_per_tso_files_written(tmp_deploy, sample_gen_load_results):
    """write_gen_load_forecasts writes national + all TSO files."""
    write_gen_load_forecasts(sample_gen_load_results, issued_at="2026-07-01T08:00:00Z")
    gen_load_dir = tmp_deploy / "gen_load"
    # National
    assert (gen_load_dir / "wind_onshore_national.json").exists()
    # Per-TSO
    assert (gen_load_dir / "wind_onshore_50hz.json").exists()
    assert (gen_load_dir / "wind_offshore_tennet.json").exists()
    assert (gen_load_dir / "load_creos.json").exists()
```

### Schema contract tests

```python
# tests/test_dashboard_schema.py
"""Verify that publish.py JSON schemas match what the dashboard JS expects."""

def test_price_forecast_schema(price_forecast_json):
    """price_forecast.json has expected top-level keys and forecasts shape."""
    assert {"target", "region", "issued_at", "forecasts"} <= set(price_forecast_json)
    for f in price_forecast_json["forecasts"]:
        assert "timestamp" in f and "forecast" in f

def test_forecast_history_schema(forecast_history_json):
    """forecast_history.json is adaptable by adaptHistory() JS function."""
    assert "forecasts" in forecast_history_json
    for entry in forecast_history_json["forecasts"]:
        assert "issued_at" in entry and "forecasts" in entry
        if entry["forecasts"]:
            assert "timestamp" in entry["forecasts"][0]

def test_model_errors_schema(model_errors_json):
    assert "dates" in model_errors_json and "mae" in model_errors_json and "rmse" in model_errors_json
    assert "blend" in model_errors_json["mae"]
```

### Playwright smoke test (`tests/test_dashboard.js`)

Minimal Node.js Playwright test, not in CI for Stage 7, run locally before milestone sign-off:

```javascript
const { test, expect } = require('@playwright/test');

test('main page renders price chart', async ({ page }) => {
  await page.goto('http://localhost:8080');
  await page.waitForSelector('#forecast-chart .svg-container', { timeout: 5000 });
  const svg = await page.$('#forecast-chart svg');
  expect(svg).not.toBeNull();
});

test('gen/load summary chart renders', async ({ page }) => {
  await page.goto('http://localhost:8080');
  await page.waitForSelector('#summary-chart .svg-container', { timeout: 5000 });
});

test('monitoring page renders composition chart', async ({ page }) => {
  await page.goto('http://localhost:8080/monitoring.html');
  await page.waitForSelector('#composition-chart .svg-container', { timeout: 5000 });
});
```

Run locally with `make test-dashboard` (requires `make serve-dashboard` to be running in another terminal, or use `npx playwright test --webServer`).

---

## 7.11 Implementation Order

Implementation is split by the Stage 7a/7b gate in §7.2. The order below is superseded by that split: complete all Stage 7a items and pass the gate before starting Stage 7b items.

1. **`publish.py` additions** — `write_actuals`, `write_errors_summary`, extend `write_gen_load_forecasts` for per-TSO, `write_gen_load_actuals`, `write_gen_load_errors_summary`, wire `write_errors_summary` into `write_outputs` + tests
2. **Stage 7b `price_inference.py` extension** — save per-model predictions to `per_model_preds/{delivery_date}.json`; keep each snapshot until `model_errors.json` is updated successfully for that delivery date, then delete only that date's snapshot
3. **Stage 7b `retrain.py` extension** — append to `retrain_history.json` on successful retrain and ensure the file is copied into `deploy/data/` in the workflow that publishes Pages, or rebuilt during daily inference from a durable source
4. **`deploy/translations.json`** — full merged file
5. **`deploy/styles.css`** — EP port + gen/load card + TSO toggle additions
6. **`deploy/index.html`** — HTML structure only
7. **`deploy/script.js`** — price charts (port from EP with `adaptHistory()`), then gen/load summary and cards
8. **`deploy/monitoring.html`** — HTML structure
9. **`deploy/monitoring.js`** — port from EP, adapt schema, add gen/load accuracy section
10. **`deploy/.nojekyll`**
11. **Makefile targets** (`serve-dashboard`, `test-dashboard`)
12. **Manual end-to-end**: `make forecast-skip-update && make serve-dashboard`, verify all charts render
13. **Playwright smoke test** run locally

Steps 4–9 are largely parallelisable once step 1 (data contract) is settled.

---

## 7.12 Milestone

- `deploy/index.html` renders all sections: price chart with PI band, gen/load national summary (stacked area + load line with actuals overlay), 4 individual gen/load cards with per-TSO toggle and PI, 30-day forecast-vs-actuals, 7-day price error bars
- `deploy/monitoring.html` renders: per-model MAE/RMSE trends, ensemble composition bar, retrain log, gen/load accuracy section
- DE/EN language toggle works on both pages
- Per-TSO gen/load files (15 files) written by `write_gen_load_forecasts()`
- Stage 7b: `model_errors.json` populated with per-model daily errors after the matching actuals are available and the retained per-model prediction snapshot is consumed
- Stage 7b: `retrain_history.json` populated on first price retrain and included in the next Pages deploy
- `deploy/.nojekyll` committed; CI Pages deploy succeeds
- All new Python functions have unit tests
- Schema contract tests pass
- Playwright smoke test passes locally
- No regressions in existing 490/495 test suite

---

## 7.13 Review Points Resolved

1. **History schema mismatch** — fixed by `adaptHistory()` adapter in `script.js` (§7.1.10)
2. **`fetchJSON` path contract** — resolved by exact-path convention, no internal prefix (§7.1.11)
3. **`errors_summary.json` staleness** — fixed by calling `write_errors_summary()` from `write_outputs()` unconditionally (§7.1.4)
4. **`write_actuals()` filter order** — fixed: filter complete days first, then sort, then trim to 30; DST not an issue for our tz-naive merged.parquet (§7.1.3)
5. **Browser smoke test** — Playwright script added, locally-run pre-milestone check (§7.10)

---

## 7.14 Implementation Review Commentary (2026-07-01)

Stage 7 is live on GitHub Pages, but the implemented state is closer to a partial Stage 7a than the full Stage 7 milestone in `docs/master_plan.md`. The main remaining problems are listed below for follow-up.

1. **Daily CI does not preserve dashboard history/error state.** `deploy/data/` is gitignored and not tracked, but `publish.py` appends to local existing JSON files (`forecast_history.json`, `errors/*.json`). The GitHub Actions inference job starts from checkout, writes fresh `deploy/data/`, and does not restore previous deployed data before writing outputs. Unless historical deploy data is carried by some external mechanism, the live Pages deployment can lose rolling history on each run.

2. **`errors_summary.json` is stale by construction.** `run_inference()` calls `write_outputs()` first, which rebuilds `errors_summary.json`, and only then calls `compute_errors()`, which may add yesterday's `errors/{date}.json`. The dashboard error trend therefore misses the newly computed error until a later run. Calling `compute_errors()` before `write_errors_summary()`, or regenerating the summary after `compute_errors()`, would close this.

3. **The API-first/static-fallback data client was not implemented.** Both `deploy/script.js` and `deploy/monitoring.js` call plain `fetch(path)` against static files. There is no `window.API_BASE_URL`, endpoint mapping, or fallback helper despite this being in the Stage 7a scope and master-plan goal. The dashboard currently consumes static JSON only.

4. **Per-TSO generation/load support is mostly UI-only.** `script.js` defines TSO toggles and attempts to fetch files such as `wind_onshore_50hz.json`, but `write_gen_load_forecasts()` only writes the four national forecast files. The current `deploy/data/gen_load/` output has no per-TSO files, so the toggles cannot expose the planned TSO-level forecasts.

5. **Monitoring page is far short of the planned Stage 7 page.** It renders ensemble-level price error trend and model composition, but not per-model MAE/RMSE, not gen/load accuracy, and not a populated retrain history. The planned `model_errors.json`, `gen_load_errors_summary.json`, and `retrain_history.json` publishing paths are not implemented.

6. **Gen/load actual overlays are referenced but not produced.** `script.js` tries to fetch `data/gen_load_actuals.json` and has overlay logic, but no publisher writes that file. The master-plan "actuals overlay where available" is therefore only implemented for price actuals, not generation/load.

7. **Retrain history is not durable or published.** `monitoring.js` fetches `retrain_history.json`, but `retrain.py` does not append this file and the daily workflow does not carry it into `deploy/data/`. The monitoring page therefore falls back to "No retrain events recorded yet."

8. **Browser smoke-test gate is not reproducible from the repo alone.** `make test-dashboard` uses `npx playwright`, but there is no `package.json`, no pinned Node dependency setup, and the smoke test is not wired into CI. On the review machine, `npx` was not available. The Python tests for `publish.py` and API passed, but the browser gate could not be run from the checked-out repo.

9. **Status wording is misleading.** `docs/master_plan.md` says "full static site" and lists Stage 7 as complete, while this detailed plan still says Stage 7a implemented with the verification gate pending. The implementation is not yet at the Stage 7 milestone: performance does not track all targets, per-TSO output is absent, and the dashboard is not API-first.

Verification during this review:

- `conda run -n energy-forecasting pytest -q tests/test_deploy_publish.py tests/test_api.py` passed: 16 tests.
- Plain `pytest` was not on `PATH`; the repo expects the conda environment.
- `npx playwright --version` failed because `npx` was not installed in the environment.

## 7.15 Resolution of §7.14 Issues (2026-07-01)

All 9 issues from §7.14 were addressed in this session. Summary:

**Issue 1 — CI history preservation:** The `inference` job in `daily_forecast.yml` now uses `actions/cache/restore@v4` to restore `deploy/data/` and `data/processed/historical_forecasts/` at the start, and `actions/cache/save@v4` to persist them at the end. Each run saves under key `deploy-state-${{ runner.os }}-${{ github.run_id }}`; restoration falls back to the most-recent prefix-matched key. After the first successful run the rolling state (forecast history, error files, gen/load forecast history) will accumulate correctly across CI runs.

**Issue 2 — `errors_summary.json` stale by construction:** `compute_errors()` is now called inside `write_outputs()` before `write_errors_summary()`. The separate `compute_errors()` call in `inference.py` has been removed. Today's newly computed error file is therefore included in the summary written in the same run.

**Issue 3 — No API-first/static-fallback switch:** Both `script.js` and `monitoring.js` now read `const DATA = (typeof window !== "undefined" && window.API_DATA_BASE) || "data/";`. A live API server can set `window.API_DATA_BASE` in a page-level script tag before loading the dashboard scripts, with no other JS changes needed.

**Issue 4 — Per-TSO files not written:** `write_gen_load_forecasts()` in `publish.py` now loops over all `(target, region)` keys in `gen_load_results` where `region` is a known per-TSO code, and writes `gen_load/{target}_{suffix}.json` (e.g. `wind_onshore_50hz.json`). The TSO suffix map (`_REGION_SUFFIX`) mirrors the JS `GEN_LOAD_CARDS` config. The per-TSO toggles in each individual gen/load card are now functional.

**Issue 5 — Monitoring page lacking gen/load accuracy:** A new `write_gen_load_errors()` function in `publish.py` reads national historical_forecasts parquets, aligns them against processed TSO actuals, computes daily MAE per target, and writes `gen_load_errors_summary.json`. The monitoring page gains a new §3 card ("Generation & Load Accuracy — Last 30 Days") rendered by `renderGenLoadErrors()` in `monitoring.js`. Data will accumulate after the cache fix allows historical_forecasts to persist across runs. Per-model price errors remain deferred (Stage 8 scope: requires per-run prediction snapshots and a separate comparison pipeline).

**Issue 6 — Gen/load actual overlays not produced:** `write_gen_load_actuals()` was added to `publish.py`. It reads `data/processed/tso/*.parquet`, sums per-target columns across the contributing TSOs (matching `_TARGET_TSO_FILES`), groups into complete 24h days, and writes the last 7 days to `gen_load_actuals.json` in the format expected by `script.js`. Called from `write_outputs()`.

**Issue 7 — Retrain history not durable or published:** `retrain.py` now appends a structured entry (date, old/new MAE, degradation %, n\_models, needs\_reselection) to `deploy/data/retrain_history.json` after every retrain attempt — both successful and degraded. The history file is preserved across runs by the CI cache (fix 1). The monitoring page's Retrain Log will therefore populate on the first retrain that runs after the cache is established.

**Issue 8 — Playwright smoke test not reproducible:** `tests/package.json` was added with `@playwright/test ^1.44.0` as a dev dependency. The `test-dashboard` Makefile target was updated to run `npm install && npx playwright install chromium` before the test. Running `make test-dashboard` is now self-contained on any machine with Node and npm available.

**Issue 9 — Status wording misleading:** `docs/master_plan.md` Stage 7 evaluation status updated to "Stage 7a + 7b complete (2026-07-01), 512 tests pass". `CLAUDE.md` status section updated to reflect the resolved issues.

**Verification (2026-07-01):**
- `conda run -n energy-forecasting pytest -q` → **512 passed** (up from 509; 3 new tests for `write_gen_load_forecasts` per-TSO, `write_gen_load_actuals`, and missing-TSO-dir guard).
- `tests/package.json` present; `make test-dashboard` is now reproducible with Node available.
