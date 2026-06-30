"""Merge pipeline configuration constants.

Key dates for bidding-area and pricing-regime transitions,
column name references for target price creation, and
imputation thresholds.
"""

import pandas as pd

# -- Regime-change timestamps (UTC) ----------------------------------------
# Germany-Austria-Luxembourg -> Germany-Luxembourg bidding area split.
# Exact boundary: 2018-10-01 00:00 CET = 2018-09-30 22:00 UTC.
BIDDING_AREA_SPLIT = pd.Timestamp("2018-09-30T22:00:00", tz="UTC")

# EPEX SPOT resolution changes from hourly to 15-minute intervals.
QUARTER_HOURLY_START = pd.Timestamp("2025-10-01", tz="UTC")

# -- Column names for unified target price ---------------------------------
# Must match output of clean_column_name() applied to SMARD descriptions.
# Filter 4169: "Marktpreis: Deutschland/Luxemburg"
PRICE_POST_SPLIT = "marktpreis_deutschland_luxemburg"
# Filter 251: "Marktpreis: Deutschland/Oesterreich/Luxemburg"
PRICE_PRE_SPLIT = "marktpreis_deutschland_oesterreich_luxemburg"

# -- Periodicity enforcement -----------------------------------------------
# Maximum consecutive missing hourly timestamps before raising an error.
MAX_MISSING_CONSECUTIVE = 3

# -- Imputation gap thresholds ---------------------------------------------
# Three tiers:
#   1. Small (<=SMALL_GAP_MAX): cubic spline interpolation (existing in clean())
#   2. Medium (SMALL_GAP_MAX < gap <= MEDIUM_GAP_MAX): same-hour-of-day averaging
#   3. Large (> MEDIUM_GAP_MAX): rejected -- logged as structural issue
SMALL_GAP_MAX = 5  # hours
MEDIUM_GAP_MAX = 48  # hours

# Window (days) for same-hour-of-day imputation.
IMPUTE_WINDOW_DAYS = 14

# -- SMARD physical limit warnings -----------------------------------------
# Pre-cleaning diagnostic: values outside these bounds are logged as
# warnings (not clipped or removed). Uses fnmatch patterns.
# Bounds are in MW except prices in EUR/MWh.
SMARD_WARN_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "stromerzeugung_*": (0, 100_000),
    "stromverbrauch_*": (20_000, 100_000),
    "prognostiziert*": (0, 120_000),
    "target_price": (-500, 1_000),
    "cross-border_flows_*": (None, 50_000),
}

# -- TSO aggregation mapping -----------------------------------------------
# Maps per-TSO filter key names to the equivalent national column name.
# Used for cross-validation (stage 3) and potential aggregation (future).
TSO_TO_NATIONAL: dict[str, str] = {
    "wind_offshore": "stromerzeugung_wind_offshore",
    "wind_onshore": "stromerzeugung_wind_onshore",
    "solar": "stromerzeugung_photovoltaik",
    "load": "stromverbrauch_gesamt_(netzlast)",
    "biomass": "stromerzeugung_biomasse",
    "gas": "stromerzeugung_erdgas",
    "hard_coal": "stromerzeugung_steinkohle",
    "lignite": "stromerzeugung_braunkohle",
    "pumped_storage": "stromerzeugung_pumpspeicher",
    "hydro": "stromerzeugung_wasserkraft",
    "other_conv": "stromerzeugung_sonstige_konventionelle",
    "other_renew": "stromerzeugung_sonstige_erneuerbare",
}
