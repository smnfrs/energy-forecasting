"""Commodity data source configuration.

Constants for ICAP carbon allowances, Yahoo Finance commodities,
FRED gas price series, and Energy Charts fallback prices.
"""

from datetime import date

# ── Yahoo Finance tickers ───────────────────────────────────────────
TICKERS: dict[str, str] = {
    "ttf": "TTF=F",  # TTF Natural Gas Futures (EUR/MWh)
    "brent": "BZ=F",  # Brent Crude Oil Futures (USD/barrel)
    "carbon_realtime": "CO2.L",  # SparkChange EUA ETC (GBP, proxy)
}

# ── ICAP carbon system IDs ──────────────────────────────────────────
ICAP_SYSTEMS: dict[str, int] = {
    "eu_ets_phase3": 33,  # 2014-2018
    "eu_ets_phase4": 35,  # 2019-present
}

# ── Output column names ─────────────────────────────────────────────
COLUMN_NAMES: dict[str, str] = {
    "carbon": "carbon_eur_per_ton",
    "carbon_realtime": "carbon_realtime_eur_per_ton",
    "ttf": "ttf_eur_per_mwh",
    "brent": "brent_usd_per_barrel",
}

# ── Data availability start dates ───────────────────────────────────
# Used to set download start for each source. Note: Brent (BZ=F) has
# Yahoo data from ~2007, NOT just 2021 as some earlier docs suggested.
DATA_START: dict[str, date] = {
    "carbon_icap": date(2014, 11, 3),
    "carbon_realtime": date(2021, 10, 18),  # CO2.L listing date
    "ttf_yahoo": date(2017, 10, 23),  # First Yahoo TTF data
    "brent": date(2010, 1, 1),  # Yahoo has data from ~2007; pre-SMARD data useful for lags
    "fred_eu_gas": date(2002, 1, 1),  # Monthly, long history
    "fred_us_gas": date(2002, 1, 1),
}

# ── Price range validation ──────────────────────────────────────────
# (min, max) -- values outside these are flagged as suspect
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "carbon": (3.0, 200.0),  # EUR/ton
    "ttf": (5.0, 350.0),  # EUR/MWh (peaked ~339 Aug 2022)
    "brent": (10.0, 200.0),  # USD/barrel
}

# ── FRED series identifiers ─────────────────────────────────────────
FRED_SERIES: dict[str, str] = {
    "fred_eu_gas": "PNGASEUUSDM",  # EU natural gas import price (USD/MMBtu)
    "fred_us_gas": "DHHNGSP",  # US Henry Hub spot price (USD/MMBtu)
}

# ── TTF reconstruction constants ────────────────────────────────────
TTF_YAHOO_START = date(2017, 10, 23)  # First Yahoo TTF data
UNIT_CONVERSION_MMBTU_TO_MWH = 0.293  # 1 MMBtu = 0.293 MWh
TTF_RECONSTRUCTION_START = date(2014, 12, 1)  # Start of gap period

# ── Energy Charts ───────────────────────────────────────────────────
ENERGY_CHARTS_BASE_URL = "https://api.energy-charts.info"

ENERGY_CHARTS_SERIES: dict[str, dict] = {
    "da_price_de_lu": {
        "bzn": "DE-LU",
        "column": "day_ahead_price_de_lu_eur_per_mwh",
    },
}

# ── ICAP API ────────────────────────────────────────────────────────
ICAP_URL = "https://allowancepriceexplorer.icapcarbonaction.com/systems/reports/price/download"
