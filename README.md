# Energy Forecasting — predicting Germany's electricity prices from the weather

**[▶ Live dashboard](https://smnfrs.github.io/energy-forecasting/)**  ·  **[Explanation](https://smnfrs.github.io/energy-forecasting/story/forecast/)**

<!-- TODO: Add a screenshot of the live dashboard right here. For non-technical readers this is the single most useful addition — it shows the finished product in one glance. Suggested: ![Dashboard](assets/dashboard.png) -->

## What is this?

Germany runs one of Europe's largest electricity markets, and a huge share of its power now comes from wind and solar. That creates a simple but powerful link:

> **When it's windy and sunny, there's plenty of cheap renewable electricity, so prices tend to fall. When it's calm and dark, expensive backup power has to step in, and prices rise.**

This project turns that link into two forecasts for the German electricity market, produced automatically every morning from free, public weather data:

1. **How much electricity will be produced and used** over the next 7 days — split into wind, solar, and total demand.
2. **What electricity will cost** tomorrow — an hour-by-hour price forecast for the next day, with a confidence range and a plain explanation of what's driving it up or down.

Both are published to a live, self-updating dashboard, alongside a daily forecast written up in plain English.

## Why it's interesting

Electricity in Germany is sold a day ahead: every day at noon, a single auction sets the price for all 24 hours of the *next* day. Because weather drives so much of the supply, tomorrow's weather is one of the strongest clues to tomorrow's price — and weather forecasts are available days in advance.

There's a catch that makes this harder than it looks. The official electricity-production forecasts published by Germany's grid operators don't come out until around **6 p.m. — six hours *after* the auction has already closed.** So a price forecast can't wait for them. This project gets around that by producing its **own** production forecasts from weather data hours *before* the auction, then feeding those into the price forecast. In other words, one model's output becomes another model's input, timed to beat the market's own deadline.

## What this project demonstrates

For anyone evaluating the work rather than the electricity market, this project shows:

- **End-to-end machine learning** — from raw public data all the way to a finished, live product that updates on its own, not just a one-off analysis in a notebook.
- **An automated daily pipeline** — code that collects, cleans, and combines data from several public sources every morning, makes its forecasts, and publishes them, with no manual steps.
- **A real, deployed service** — a live web dashboard, a programmatic interface (API) for the data, and scheduled automation that runs unattended.
- **Honest, explainable predictions** — every forecast comes with a confidence range (how sure the model is) and a breakdown of *which factors* pushed the prediction up or down.
- **Real-world domain modelling** — encoding actual physics (how wind speed converts into power, how temperature drives heating and cooling demand) rather than treating the data as anonymous numbers.

---

## How it works

The whole system flows in one direction, from weather to prices:

```
Public weather data
      │
      ▼
Production & demand forecasts        →  Deliverable 1: 7-day forecasts of wind,
(wind, solar, demand, net balance),      solar, demand and net balance, per grid region
7 days ahead, for each grid region
      │
      ▼
Price forecast + confidence range    →  Deliverable 2: next-day, hour-by-hour
for the whole German price region        price forecast for Germany/Luxembourg
```

### Production & demand forecasting

Physics-informed features (wind power density, air density, wind shear, solar elevation, heating/cooling degree days) are computed from [Open-Meteo](https://open-meteo.com/) weather data at 30+ curated locations near real generation assets, then aggregated spatially per grid region. Forecasts are produced recursively step by step (each hour forecast off the previous one) out to 168 hours, using gradient-boosted trees (LightGBM, XGBoost) and linear models (ElasticNet) as base learners, combined with stacking ensembles. Prediction intervals come from conformal methods (MAPIE) — a way of attaching a calibrated "how confident is the model" range to each point.

Training order is enforced by dependency: wind/solar first, then demand (which uses wind/solar actuals), then the net generation-minus-load balance (which uses everything). Outputs are published per grid region and as a national total.

> **Grid regions:** Germany's transmission network is divided into control areas, each operated by a different company (50Hertz, Amprion, TenneT, TransnetBW, and Creos in Luxembourg). Forecasts are produced for each area and summed to a national figure.

### Price forecasting

Market data from [SMARD](https://www.smard.de/home) (electricity generation by source, consumption, cross-border flows, historical prices) is combined with commodity prices (EU carbon allowances, natural gas, Brent crude) and this project's own production/demand forecasts, then turned into an engineered feature set: lagged price statistics, moving averages, generation-mix percentages, a supply-versus-demand gap, and cross-border aggregates.

A weighted blend of several models (LightGBM, two XGBoost variants, CatBoost, and a Ridge linear model — blend weights fit on held-out data via inverse-error weighting) produces the next-day, hour-by-hour price forecast. Each forecast ships with a conformal prediction interval and a per-hour driver attribution (using SHAP, a method that quantifies how much each input pushed the prediction up or down).

### Deployment

Both forecasts are served from a stateless FastAPI service, a static dashboard, and daily/monthly GitHub Actions automation.

## Data sources

- **[SMARD](https://www.smard.de/home)** (German Federal Network Agency) — per-region and national electricity generation, consumption, cross-border flows, and day-ahead prices. Hourly, Dec 2014–present. No API key required.
- **[Open-Meteo](https://open-meteo.com/)** — historical, recent-forecast, and future-forecast weather (wind speed/direction, temperature, solar radiation, precipitation) stitched into continuous series.
- **Commodities** (price model only) — EU carbon allowances, natural gas futures, and Brent crude.

## Production deployment

- **Command-line tool** (`energy-forecasting`) — `download` / `update` / `train` / `deploy` command groups covering the full pipeline from raw data to served forecast, for both products.
- **API** — FastAPI with health, price forecast, production/demand forecast, forecast history, and model metadata endpoints.
- **Dashboard** — static site (`deploy/`) showing price plus confidence band, national and per-region production/demand, history overlays, error bands, a German/English toggle, and a monitoring page.
- **Story site** (`deploy/story/`) — a scrollable history of the German energy market, plus a daily page that combines the model's driver attribution with an automatically written plain-English summary.
- **Scheduled automation** — a daily job (data update → production forecast → price forecast → publish) and a scheduled retrain job.

## Project structure

```
├── energy_forecasting/     # Source package
│   ├── config/             # Paths, constants, search spaces, data/location config
│   ├── data/               # Data acquisition + merge/processing
│   ├── features/           # Physics-informed and market feature engineering
│   ├── modeling/           # Training, tuning, cross-validation, ensembling, metrics
│   ├── deploy/             # Inference, retrain, publish, explanations, narrative, monitoring
│   ├── api/                # FastAPI app, routes, schemas
│   └── cli.py              # `energy-forecasting` entry point
│
├── deploy/                 # Static dashboard + story site (served via GitHub Pages)
├── data/                   # Data directory (gitignored)
├── models/                 # Tracked model artifacts, production models
├── dev-notes/              # Stage plans, conventions, roadmap
├── tests/
├── .github/workflows/      # daily_forecast, retrain, deploy_static, backfill
├── Makefile
└── pyproject.toml
```

See `dev-notes/master_plan.md` for the full staged build-out and `dev-notes/source_repo_guide.md` for where functionality originated in the two source repos.

## Setup

**Prerequisites:** Python 3.13, conda.

```bash
make env               # create conda environment
conda activate energy-forecasting
make install           # install package + dev dependencies
```

## Usage

```bash
# Data
make data              # full historical download
make update            # incremental update

# Feature engineering
make features-full

# Training — production/demand must precede price
make train-gen-load
energy-forecasting train price

# Inference — both forecasts (mirrors the daily automated run)
make forecast

# Serve API + dashboard locally
make serve
make serve-dashboard
```

Run `make help` for the full target list, `energy-forecasting --help` for the command-line tool, and `make test` for the test suite.

## Limitations

Uses exclusively free, public data sources, so it's missing the richer market data a trading desk would have — order-book depth, bidding volumes, futures curves, and intraday prices. Commodity prices (carbon, gas, oil) are reconstructed daily closing prices rather than full trading-day or futures-curve data. See `dev-notes/roadmap.md` for planned improvements.

## Acknowledgements

This project merges and builds on two earlier repositories:

- **[energy_prices](https://github.com/smnfrs/energy-prices)** — day-ahead price forecasting, feature engineering, and the blended-ensemble methodology.
- **[energy_market_analysis](https://github.com/smnfrs/energy_market_analysis)**, itself a fork of [vsevolodnedora/energy_market_analysis](https://github.com/vsevolodnedora/energy_market_analysis) by [Vsevolod Nedora](https://github.com/vsevolodnedora) — the curated weather locations, physics-informed feature engineering, spatial aggregation, and multi-step recursive forecasting architecture originate there. See the [upstream README](https://github.com/vsevolodnedora/energy_market_analysis#readme) and the author's [Medium articles](https://medium.com/@vsevolod.nedora).

Thanks also to *Modern Time Series Forecasting with Python* (Manu Joseph) and the instructors at WBS Coding School.

## License

MIT
