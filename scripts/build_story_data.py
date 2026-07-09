"""Precompute static JSON datasets for the Stage 9 storytelling site.

Reads data/processed/merged.parquet and writes one small JSON file per chart
into deploy/story/data/. Run manually (`make story-data`) — this is historical,
slow-moving narrative data, not the daily forecast pipeline.
"""

import json
from pathlib import Path

import pandas as pd

MERGED_PATH = Path("data/processed/merged.parquet")
OUT_DIR = Path("deploy/story/data")
HTML_PATHS = [Path("deploy/story/chart_prototypes.html"), Path("deploy/story/index.html")]

_REGISTRY: dict = {}

NEIGHBOUR_COLS = {
    "AT": "marktpreis_oesterreich",
    "FR": "marktpreis_frankreich",
    "NL": "marktpreis_niederlande",
    "PL": "marktpreis_polen",
}

# fixed stacking order: renewables first (anchored at zero), then nuclear, then fossil, then other
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
RENEWABLE_GROUPS = ["wind", "solar", "hydro_biomass"]

EXAMPLE_DAY = "2025-05-11"


def write_json(name: str, payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=None, separators=(",", ":"), allow_nan=False))
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    _REGISTRY[name] = payload


def inline_into_html() -> None:
    """Embed all datasets into each page in HTML_PATHS so they work via file://
    (fetch() to a local JSON file is blocked by browsers, <script> data isn't)."""
    start_marker = "<!-- STORY_DATA_START -->"
    end_marker = "<!-- STORY_DATA_END -->"
    blob = json.dumps(_REGISTRY, indent=None, separators=(",", ":"), allow_nan=False)
    for html_path in HTML_PATHS:
        text = html_path.read_text()
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker)
        text = text[:start] + "\n" + blob + "\n" + text[end:]
        html_path.write_text(text)
        print(f"inlined {len(_REGISTRY)} datasets into {html_path}")


def nanlist(s: pd.Series, decimals: int) -> list:
    """Round a series to a plain list, replacing NaN with None (json.dumps writes
    bare NaN, which is not valid JSON and JSON.parse rejects it)."""
    return [None if pd.isna(v) else round(float(v), decimals) for v in s]


def build_price_history(df: pd.DataFrame) -> None:
    monthly = df["target_price"].resample("MS").agg(["mean", "min", "max"])
    write_json(
        "price_history.json",
        {
            "month": [d.strftime("%Y-%m") for d in monthly.index],
            "mean": nanlist(monthly["mean"], 2),
            "min": nanlist(monthly["min"], 2),
            "max": nanlist(monthly["max"], 2),
        },
    )


def build_gas_shock(df: pd.DataFrame) -> None:
    monthly_gas = df["ttf_eur_per_mwh"].resample("MS").mean()
    monthly_price = df["target_price"].resample("MS").mean()
    write_json(
        "gas_shock.json",
        {
            "month": [d.strftime("%Y-%m") for d in monthly_gas.index],
            "gas_ttf_eur_mwh": nanlist(monthly_gas, 2),
            "power_price_eur_mwh": nanlist(monthly_price, 2),
        },
    )


def build_bidding_zones(df: pd.DataFrame) -> None:
    monthly_de = df["target_price"].resample("MS").mean()
    out = {
        "month": [d.strftime("%Y-%m") for d in monthly_de.index],
        "DE_LU": nanlist(monthly_de, 2),
    }
    for code, col in NEIGHBOUR_COLS.items():
        s = df[col].resample("MS").mean().reindex(monthly_de.index)
        out[code] = nanlist(s, 2)
    write_json("bidding_zones.json", out)


def build_generation_mix(df: pd.DataFrame) -> None:
    monthly = {
        group: df[cols].sum(axis=1).resample("MS").mean()
        for group, cols in GENERATION_GROUPS.items()
    }
    idx = next(iter(monthly.values())).index
    total = sum(monthly.values())
    renewable_share = sum(monthly[g] for g in RENEWABLE_GROUPS) / total * 100
    out = {"month": [d.strftime("%Y-%m") for d in idx]}
    for group, s in monthly.items():
        out[group] = nanlist(s, 1)
    out["renewable_share_pct"] = nanlist(renewable_share, 1)
    write_json("generation_mix.json", out)


def build_negative_prices(df: pd.DataFrame) -> None:
    neg = df["target_price"] < 0
    by_year = neg.groupby(df.index.year).sum()
    write_json(
        "negative_prices.json",
        {"year": [int(y) for y in by_year.index], "hours": [int(v) for v in by_year.values]},
    )

    day = df.loc[EXAMPLE_DAY]
    gen_cols = {
        "solar": "stromerzeugung_photovoltaik",
        "wind_onshore": "stromerzeugung_wind_onshore",
        "wind_offshore": "stromerzeugung_wind_offshore",
        "load": "stromverbrauch_gesamt_(netzlast)",
    }
    example = {
        "date": EXAMPLE_DAY,
        "hour": list(range(24)),
        "price": nanlist(day["target_price"], 2),
    }
    for name, col in gen_cols.items():
        example[name] = nanlist(day[col], 1)
    write_json("negative_price_example_day.json", example)


def main() -> None:
    df = pd.read_parquet(MERGED_PATH)
    build_price_history(df)
    build_gas_shock(df)
    build_bidding_zones(df)
    build_generation_mix(df)
    build_negative_prices(df)
    inline_into_html()


if __name__ == "__main__":
    main()
