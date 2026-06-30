"""Feature engineering constants and feature list definitions.

All feature lists use SHORT_NAMES (from config/columns.py) exclusively.
Feature strings follow the suffix DSL grammar (see docs/stage4_feature_engineering.md §4.3).
"""

from datetime import date

# ── Temporal constants ────────────────────────────────────────────

CYCLICAL_PERIODS: dict[str, int] = {
    "hour": 24,
    "day_of_week": 7,
    "month": 12,
}

DAY_INDEX_EPOCH = date(2015, 1, 5)  # Monday in first SMARD week
YEAR_INDEX_BASE = 2015

# ── Population weights for holiday calculation ────────────────────
# From Destatis, rounded to nearest thousand.

GERMAN_STATE_POPULATIONS: dict[str, int] = {
    "BW": 11_103_000,
    "BY": 13_177_000,
    "BE": 3_664_000,
    "BB": 2_537_000,
    "HB": 680_000,
    "HH": 1_853_000,
    "HE": 6_293_000,
    "MV": 1_611_000,
    "NI": 8_003_000,
    "NW": 17_926_000,
    "RP": 4_098_000,
    "SL": 983_000,
    "SN": 4_057_000,
    "ST": 2_181_000,
    "SH": 2_911_000,
    "TH": 2_120_000,
}

# ── Generation columns for total/pct computation ─────────────────
# These are the cleaned SMARD column names.

GENERATION_COLUMNS: list[str] = [
    "stromerzeugung_biomasse",
    "stromerzeugung_braunkohle",
    "stromerzeugung_erdgas",
    "stromerzeugung_kernenergie",
    "stromerzeugung_photovoltaik",
    "stromerzeugung_pumpspeicher",
    "stromerzeugung_sonstige_erneuerbare",
    "stromerzeugung_sonstige_konventionelle",
    "stromerzeugung_steinkohle",
    "stromerzeugung_wasserkraft",
    "stromerzeugung_wind_offshore",
    "stromerzeugung_wind_onshore",
]

RENEWABLE_COLUMNS: list[str] = [
    "stromerzeugung_biomasse",
    "stromerzeugung_photovoltaik",
    "stromerzeugung_sonstige_erneuerbare",
    "stromerzeugung_wasserkraft",
    "stromerzeugung_wind_offshore",
    "stromerzeugung_wind_onshore",
]

# ── Neighbour prices ──────────────────────────────────────────────
# Cleaned SMARD column names for neighbouring bidding zone prices.

NEIGHBOUR_PRICES: list[str] = [
    "marktpreis_frankreich",
    "marktpreis_niederlande",
    "marktpreis_oesterreich",
    "marktpreis_schweiz",
    "marktpreis_tschechien",
    "marktpreis_daenemark_1",
    "marktpreis_daenemark_2",
    "marktpreis_belgien",
    "marktpreis_polen",
    "marktpreis_norwegen_2",
    "marktpreis_schweden_4",
]

# ── Cross-border flow pairs ──────────────────────────────────────
# (export_column, import_column, country_short)

FLOW_PAIRS: list[tuple[str, str, str]] = [
    # The third element is the country slug used in the derived column
    # name (``_derived_net_export_{slug}``). German slugs match the
    # convention used by ``compute_price_spreads`` and ``SHORT_NAMES``;
    # keep them aligned or the engine will fail to resolve short names.
    (
        "cross-border_flows_austria_exports",
        "cross-border_flows_austria_imports",
        "oesterreich",
    ),
    (
        "cross-border_flows_belgium_exports",
        "cross-border_flows_belgium_imports",
        "belgien",
    ),
    (
        "cross-border_flows_czech_republic_exports",
        "cross-border_flows_czech_republic_imports",
        "tschechien",
    ),
    (
        "cross-border_flows_denmark_1_exports",
        "cross-border_flows_denmark_1_imports",
        "daenemark_1",
    ),
    (
        "cross-border_flows_denmark_2_exports",
        "cross-border_flows_denmark_2_imports",
        "daenemark_2",
    ),
    (
        "cross-border_flows_france_exports",
        "cross-border_flows_france_imports",
        "frankreich",
    ),
    (
        "cross-border_flows_netherlands_exports",
        "cross-border_flows_netherlands_imports",
        "niederlande",
    ),
    (
        "cross-border_flows_norway_2_exports",
        "cross-border_flows_norway_2_imports",
        "norwegen_2",
    ),
    (
        "cross-border_flows_poland_exports",
        "cross-border_flows_poland_imports",
        "polen",
    ),
    (
        "cross-border_flows_sweden_4_exports",
        "cross-border_flows_sweden_4_imports",
        "schweden_4",
    ),
    (
        "cross-border_flows_switzerland_exports",
        "cross-border_flows_switzerland_imports",
        "schweiz",
    ),
]

# ── Feature lists ─────────────────────────────────────────────────
# Short-name feature strings using the suffix DSL grammar.
# Matches EP's preprocessor_v5_slim_hourly (morning_cutoff_cet=10).

# Exact match of EP's preprocessor_v5_slim_hourly output (83 features).
# Reconstructed by tracing all 12 pipeline phases including Phase 11 drops.
PRICE_FEATURES_SLIM: list[str] = [
    # ── Temporal (7) ──
    "hour_of_day",  # raw integer 0-23
    "day_of_week",  # raw integer 0-6
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    # ── Forecasts — raw hourly, day-ahead available (8) ──
    "prog_gen_total",
    "prog_gen_wind_pv",
    "prog_gen_wind_on",
    "prog_gen_wind_off",
    "prog_gen_solar",
    "prog_gen_other",
    "prog_load",
    "prog_residual",
    # ── Forecast daily aggregate (1, EP keeps only this one) ──
    "prog_gen_wind_pv_daily_max",
    # ── Prognosticated percentages (2) ──
    "pct_prog_other",
    "pct_prog_wind_pv",
    # ── Target price rolling stats (9) ──
    "price_d7",  # D-7 single day avg
    "price_d7_d1_avg",  # 7-day rolling avg
    "price_d7_d1_std",
    "price_d7_d1_max",
    "price_d7_d1_min",
    "price_d30_d1_avg",  # 30-day rolling avg
    "price_d30_d1_std",
    "price_d2_d1_std",  # 48h volatility
    "price_d3_d1_std",  # 72h volatility
    # ── Price ranges (2) ──
    "price_d7_d1_range",
    "price_d30_d1_range",
    # ── EWMA prices — end-of-D-1 cutoff (6) ──
    "price_ewma_6_d1",
    "price_ewma_24_d1",
    "price_ewma_2160_d1",
    "price_fr_ewma_6_d1",
    "price_fr_ewma_24_d1",
    "price_fr_ewma_2160_d1",
    # ── EWMA prices — D-1 morning cutoff h7 (3) ──
    "price_ewma_6_d1_h7",
    "price_ewma_24_d1_h7",
    "price_ewma_2160_d1_h7",
    # ── EWMA actuals — D-1 morning cutoff h7 (3) ──
    "residual_load_ewma_24_d1_h7",
    "gen_wind_on_ewma_168_d1_h7",
    "gen_solar_ewma_2160_d1_h7",
    # ── EWMA commodities — D-2 end-of-day cutoff (3) ──
    "carbon_ewma_24_d2",
    "ttf_ewma_24_d2",
    "ttf_ewma_720_d2",
    # ── Morning actuals — D-1 hours 0-6 mean (3) ──
    "residual_load_d1_eh7",
    "gen_wind_on_d1_eh7",
    "gen_wind_off_d1_eh7",
    # ── Same-hour price lags (4) ──
    "price_h24",  # D-1
    "price_h48",  # D-2
    "price_h168",  # D-7
    "price_h336",  # D-14
    # ── Same-hour neighbour price lags (2) ──
    "price_fr_h24",
    "price_ch_h24",
    # ── Same-hour generation lags — D-2 (3) ──
    "gen_wind_on_h48",
    "gen_wind_off_h48",
    "gen_solar_h48",
    # ── Actuals D-2 daily mean (5) — EP overwrites raw columns to D-2 mean ──
    "gen_wind_on_d2",
    "gen_wind_off_d2",
    "gen_solar_d2",
    "gen_gas_d2",
    "load_d2",
    # ── Neighbour prices D-1 daily mean (4) — EP overwrites to D-1 mean ──
    "price_fr_d1",
    "price_nl_d1",
    "price_at_d1",
    "price_ch_d1",
    # ── Cross-border D-2 daily mean (2) ──
    "total_exports_d2",
    "total_imports_d2",
    # ── Commodities D-2 daily mean (3) — EP overwrites to D-2 mean ──
    "ttf_d2",
    "brent_d2",
    "carbon_d2",
    # ── Generation percentages D-2 (6) — EP computes on overwritten D-2 data ──
    "gen_pct_gas_d2",
    "gen_pct_pumped_d2",
    "gen_pct_hydro_d2",
    "gen_pct_other_d2",
    "gen_pct_wind_off_d2",
    "supply_demand_gap_d2",
    # ── Time indices (2) ──
    "day_index",
    "year_index",
    # ── Interaction terms (5) ──
    "prog_residual__x__day_index",
    "price_ewma_6_d1__x__day_index",
    "ttf_ewma_720_d2__x__day_index",
    "ttf_ewma_24_d2__x__day_index",
    "prog_gen_other__x__day_index",
]

PRICE_FEATURES_FULL: list[str] = [
    *PRICE_FEATURES_SLIM,
    # ── Additional temporal ──
    "hour_fourier_24_3",
    "month_sin",
    "month_cos",
    # ── Additional price rolling stats ──
    "price_d14_d1_avg",
    "price_d14_d1_std",
    "price_d7_d1_h8_h20_avg",  # peak hours
    "price_d30_d1_max",
    "price_d30_d1_min",
    # ── Additional price lags ──
    "price_h25",  # adjacent-hour D-1
    "price_h26",
    # ── Additional EWMA ──
    "price_ewma_168_d1",
    "gen_wind_on_ewma_24_d1_h7",
    "gen_solar_ewma_24_d1_h7",
    "residual_load_ewma_168_d1_h7",
    "residual_load_ewma_2160_d1_h7",
    "gen_wind_on_ewma_2160_d1_h7",
    "gen_solar_ewma_168_d1_h7",
    "carbon_ewma_720_d2",
    "carbon_ewma_2160_d2",
    "ttf_ewma_2160_d2",
    "brent_ewma_24_d2",
    "brent_ewma_720_d2",
    "brent_ewma_2160_d2",
    # ── Additional neighbour prices ──
    # price_nl_d1 / price_at_d1 already in SLIM
    "price_dk1_d1",
    "price_dk2_d1",
    "price_be_d1",
    "price_pl_d1",
    "price_cz_d1",
    # ── Additional actuals D-2 ──
    "gen_lignite_d2",
    "gen_coal_d2",
    "gen_nuclear_d2",
    "gen_biomass_d2",
    "gen_hydro_d2",
    "gen_pumped_d2",
    "gen_other_d2",
    "gen_other_renew_d2",
    "residual_load_d2",
    # ── Additional morning actuals ──
    "gen_solar_d1_eh7",
    "gen_gas_d1_eh7",
    "gen_lignite_d1_eh7",
    "gen_nuclear_d1_eh7",
    "load_d1_eh7",
    # ── Additional generation percentages ──
    # gen_pct_wind_off_d2 already in SLIM
    "gen_pct_wind_on_d2",
    "gen_pct_solar_d2",
    "gen_pct_nuclear_d2",
    "gen_pct_lignite_d2",
    "gen_pct_coal_d2",
    "pct_renewable_d2",
    # ── Commodity rolling ──
    "ttf_d7_d2_avg",
    "brent_d7_d2_avg",
    "carbon_d7_d2_avg",
    # ── Additional interactions ──
    "gen_wind_on_d1_eh7__x__day_index",
    "gen_solar_d1_eh7__x__day_index",
    "price_d7_d1_std__x__is_weekend",
]

# Superset of PRICE_FEATURES_FULL with all features the feature-selection pipeline
# (§5c.2) is allowed to choose from. Sized to ~350 entries — winners are selected
# automatically; this list is intentionally exhaustive, not minimal.
PRICE_FEATURES_MAX: list[str] = [
    *PRICE_FEATURES_FULL,
    # ── Neighbour price EWMAs (DK1/DK2/BE/PL/CZ/NO2/SE4 × 3 spans) ────
    "price_dk1_ewma_6_d1", "price_dk1_ewma_24_d1", "price_dk1_ewma_2160_d1",
    "price_dk2_ewma_6_d1", "price_dk2_ewma_24_d1", "price_dk2_ewma_2160_d1",
    "price_be_ewma_6_d1",  "price_be_ewma_24_d1",  "price_be_ewma_2160_d1",
    "price_pl_ewma_6_d1",  "price_pl_ewma_24_d1",  "price_pl_ewma_2160_d1",
    "price_cz_ewma_6_d1",  "price_cz_ewma_24_d1",  "price_cz_ewma_2160_d1",
    "price_no2_ewma_6_d1", "price_no2_ewma_24_d1", "price_no2_ewma_2160_d1",
    "price_se4_ewma_6_d1", "price_se4_ewma_24_d1", "price_se4_ewma_2160_d1",
    # ── Neighbour price hourly lags h24/h48 (FR/CH h24 already in FULL) ──
    "price_nl_h24", "price_nl_h48",
    "price_at_h24", "price_at_h48",
    "price_dk1_h24", "price_dk1_h48",
    "price_dk2_h24", "price_dk2_h48",
    "price_be_h24", "price_be_h48",
    "price_pl_h24", "price_pl_h48",
    "price_cz_h24", "price_cz_h48",
    "price_no2_h24", "price_no2_h48",
    "price_se4_h24", "price_se4_h48",
    "price_fr_h48",
    "price_ch_h48",
    # ── Additional target price EWMA spans ──────────────────────────
    "price_ewma_48_d1",
    "price_ewma_336_d1",
    "price_ewma_720_d1",
    # ── Additional target price lags ────────────────────────────────
    "price_h72",   # D-3
    "price_h96",   # D-4
    # ── Additional target price rolling windows ─────────────────────
    "price_d14_d1_max",
    "price_d14_d1_min",
    "price_d14_d1_range",
    "price_d3_d1_avg",
    "price_d5_d1_avg",
    "price_d5_d1_std",
    "price_d7_d1_h0_h8_avg",
    "price_d7_d1_h20_h24_avg",
    # (gen_biomass_d2 / gen_nuclear_d2 / gen_pumped_d2 / gen_other_d2 already in FULL)
    # ── Additional D-2 generation h48 lags ──────────────────────────
    "gen_gas_h48", "gen_lignite_h48", "gen_coal_h48",
    "gen_nuclear_h48", "gen_biomass_h48", "gen_hydro_h48",
    "gen_pumped_h48", "gen_other_h48", "gen_other_renew_h48",
    "load_h48",
    # ── Additional morning-cutoff D-1 actuals (eh7) ─────────────────
    # gen_wind_off_d1_eh7 already in SLIM
    "gen_biomass_d1_eh7",
    "gen_pumped_d1_eh7",
    "gen_hydro_d1_eh7",
    "gen_other_d1_eh7",
    "gen_other_renew_d1_eh7",
    "gen_coal_d1_eh7",
    # ── Additional D-2 generation percentages ───────────────────────
    # gen_pct_pumped_d2 / gen_pct_other_d2 already in SLIM
    "gen_pct_biomass_d2",
    "gen_pct_other_renew_d2",
    # ── Cross-border per-country net flows D-2 ──────────────────────
    "net_export_at_d2", "net_export_be_d2", "net_export_cz_d2",
    "net_export_dk1_d2", "net_export_dk2_d2", "net_export_fr_d2",
    "net_export_nl_d2", "net_export_no2_d2", "net_export_pl_d2",
    "net_export_se4_d2", "net_export_ch_d2",
    # ── Additional commodity rolling stats ──────────────────────────
    "ttf_d14_d2_avg", "ttf_d30_d2_avg", "ttf_d30_d2_std",
    "brent_d14_d2_avg", "brent_d30_d2_avg",
    "carbon_d14_d2_avg", "carbon_d30_d2_avg", "carbon_d30_d2_std",
    "ttf_d7_d2_std", "brent_d7_d2_std", "carbon_d7_d2_std",
    # ── Additional morning-cutoff actuals EWMA ──────────────────────
    "residual_load_ewma_6_d1_h7",
    "gen_wind_off_ewma_24_d1_h7",
    "gen_wind_off_ewma_168_d1_h7",
    "load_ewma_24_d1_h7",
    "load_ewma_168_d1_h7",
    "gen_gas_ewma_24_d2",
    # ── Forecast daily aggregate variants (offset=0, no lag) ────────
    "prog_gen_total_daily_max", "prog_gen_total_daily_avg",
    "prog_load_daily_max", "prog_load_daily_avg",
    "prog_residual_daily_max", "prog_residual_daily_avg",
    "prog_gen_solar_daily_max", "prog_gen_solar_daily_sum",
    "prog_gen_wind_on_daily_max",
    # ── Per-technology prognosis percentages (new in 5c.0) ──────────
    "pct_prog_solar",
    "pct_prog_wind_on",
    "pct_prog_wind_off",
    # ── Price volatility / momentum ratios ──────────────────────────
    "price_d2_d1_std__x__price_d7_d1_std",
    "price_d7_d1_avg__x__price_d30_d1_avg",
    # ── EEG regime & negative-price rolling signal ──────────────────
    "eeg_regime",
    "neg_price_frac_30d_d1",
    "neg_price_frac_90d_d1",
    "neg_price_depth_30d_d1",
    # Stage 5b gen/load historical_forecasts: no longer separate features.
    # They overlay onto prog_gen_wind_on / prog_gen_wind_off / prog_gen_solar /
    # prog_load at dataset prep (waterfall EMA → SMARD → actuals), so the
    # existing prog_* columns and their derivatives carry the upgraded values.
    # ── Additional interaction terms ────────────────────────────────
    "price_ewma_24_d1__x__hour_sin",
    "prog_residual__x__is_weekend",
    "prog_gen_solar__x__hour_sin",
    "prog_gen_wind_pv__x__prog_load",
    "price_d7_d1_avg__x__day_index",
    "carbon_ewma_24_d2__x__gen_pct_gas_d2",
    "ttf_ewma_24_d2__x__gen_pct_gas_d2",
    "prog_residual__x__is_holiday",
    "price_d2_d1_std__x__day_index",
    "load_d1_eh7__x__day_index",
]

GEN_LOAD_FEATURES: list[str] = [
    # Generation/load models use weather features (computed at train time)
    # plus basic temporal and lagged generation/load features.
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "day_index",
    "hour_fourier_24_3",
    # Lagged actuals
    "gen_wind_on_h24",
    "gen_wind_off_h24",
    "gen_solar_h24",
    "load_h24",
    "gen_wind_on_d7_d2_avg",
    "gen_solar_d7_d2_avg",
    "load_d7_d2_avg",
]
