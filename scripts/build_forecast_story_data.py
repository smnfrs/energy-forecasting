"""Precompute rolling-trailing-365-day facts for the Stage 10 forecast story.

Reads data/processed/merged.parquet and writes deploy/story/forecast/data/facts_yearly.json:
daily generation-by-fuel-group + load series, fuel-mix %, imports/exports as %
of domestic generation, hourly price stats, all over a rolling trailing year —
plus the same blocks for the prior year (YoY) and the last 7 days vs. the
equivalent 7 days ~52 weeks back (WoW).

Run manually/periodically (`make story-forecast-data`, weekly) — this is
slow-moving yearly-recap data, not the daily forecast pipeline. Chained with
the narrative Groq call in energy_forecasting/deploy/narrative.py
(generate_yearly_narrative), which reads this file's output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MERGED_PATH = Path("data/processed/merged.parquet")
OUT_DIR = Path("deploy/story/forecast/data")
OUT_PATH = OUT_DIR / "facts_yearly.json"

LOAD_COL = "stromverbrauch_gesamt_(netzlast)"
PRICE_COL = "target_price"

# Same grouping as scripts/build_story_data.py (kept in sync manually — both
# read the raw German SMARD column names, which don't change).
GENERATION_GROUPS = {
    "wind": ["stromerzeugung_wind_onshore", "stromerzeugung_wind_offshore"],
    "solar": ["stromerzeugung_photovoltaik"],
    "hydro_biomass": ["stromerzeugung_wasserkraft", "stromerzeugung_biomasse"],
    "nuclear": ["stromerzeugung_kernenergie"],
    "gas": ["stromerzeugung_erdgas"],
    "coal": ["stromerzeugung_braunkohle", "stromerzeugung_steinkohle"],
    "other": [
        "stromerzeugung_pumpspeicher",
        "stromerzeugung_sonstige_erneuerbare",
        "stromerzeugung_sonstige_konventionelle",
    ],
}

_NEIGHBOURS = [
    "denmark_1", "denmark_2", "netherlands", "northern_italy", "switzerland",
    "czech_republic", "france", "sweden_4", "hungary", "slovenia", "belgium",
    "poland", "norway_2", "austria",
]
IMPORT_COLS = [f"cross-border_flows_{n}_imports" for n in _NEIGHBOURS]
EXPORT_COLS = [f"cross-border_flows_{n}_exports" for n in _NEIGHBOURS]


def nanlist(s: pd.Series, decimals: int) -> list:
    return [None if pd.isna(v) else round(float(v), decimals) for v in s]


def _window(df: pd.DataFrame, end: pd.Timestamp, days: int) -> pd.DataFrame:
    start = end - pd.Timedelta(days=days)
    return df.loc[(df.index > start) & (df.index <= end)]


def _gen_load_block(window: pd.DataFrame) -> dict:
    """Daily gen-by-fuel-group + load series, fuel-mix %, imports/exports %."""
    daily_gen = {
        group: window[[c for c in cols if c in window.columns]].sum(axis=1).resample("D").sum()
        for group, cols in GENERATION_GROUPS.items()
    }
    daily_load = window[LOAD_COL].resample("D").sum()

    total_gen = sum(daily_gen.values())
    mix_pct = {
        group: round(float((s.sum() / total_gen.sum()) * 100), 1)
        for group, s in daily_gen.items()
    }

    # Import columns are signed negative (inbound flow) in the raw data; take
    # the magnitude so both percentages read as positive shares of domestic generation.
    total_imports = window[[c for c in IMPORT_COLS if c in window.columns]].sum().sum()
    total_exports = window[[c for c in EXPORT_COLS if c in window.columns]].sum().sum()
    total_imports = abs(total_imports)
    domestic_gen = total_gen.sum()

    idx = daily_load.index
    return {
        "date": [d.strftime("%Y-%m-%d") for d in idx],
        "generation": {group: nanlist(s.reindex(idx), 1) for group, s in daily_gen.items()},
        "load": nanlist(daily_load.reindex(idx), 1),
        "fuel_mix_pct": mix_pct,
        "imports_pct_of_domestic_gen": round(float(total_imports / domestic_gen * 100), 2)
        if domestic_gen else None,
        "exports_pct_of_domestic_gen": round(float(total_exports / domestic_gen * 100), 2)
        if domestic_gen else None,
    }


def _price_block(window: pd.DataFrame) -> dict:
    """Hourly price series + mean/extreme-hour/negative-hour stats."""
    price = window[PRICE_COL].dropna()
    by_hour = price.groupby(price.index.hour).mean()
    return {
        "timestamp": [ts.isoformat() for ts in price.index],
        "price": nanlist(price, 2),
        "mean_price": round(float(price.mean()), 2) if len(price) else None,
        "most_expensive_hour": int(by_hour.idxmax()) if len(by_hour) else None,
        "most_expensive_hour_avg_price": round(float(by_hour.max()), 2) if len(by_hour) else None,
        "least_expensive_hour": int(by_hour.idxmin()) if len(by_hour) else None,
        "least_expensive_hour_avg_price": round(float(by_hour.min()), 2) if len(by_hour) else None,
        "negative_price_hours": int((price < 0).sum()),
    }


def build(df: pd.DataFrame) -> dict:
    end = df.index.max().normalize() + pd.Timedelta(hours=23)

    current_year = _window(df, end, 365)
    prior_year = _window(df, end - pd.Timedelta(days=365), 365)
    last_week = _window(df, end, 7)
    same_week_last_year = _window(df, end - pd.Timedelta(days=364), 7)

    return {
        "generated_through": end.isoformat(),
        "current_year": {
            "gen_load": _gen_load_block(current_year),
            "price": _price_block(current_year),
        },
        "prior_year": {
            "gen_load": _gen_load_block(prior_year),
            "price": _price_block(prior_year),
        },
        "last_7_days": {
            "gen_load": _gen_load_block(last_week),
            "price": _price_block(last_week),
        },
        "same_7_days_prior_year": {
            "gen_load": _gen_load_block(same_week_last_year),
            "price": _price_block(same_week_last_year),
        },
    }


def main() -> None:
    df = pd.read_parquet(MERGED_PATH)
    payload = build(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=None, separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
