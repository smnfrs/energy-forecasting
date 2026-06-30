"""Short name registry and SMARD filter key mappings.

The short name registry maps concise names (used in feature strings like
``price_d1``) to actual DataFrame column names (snake_case German, matching
SMARD's naming after clean_column_name).

SMARD filter key mappings are ported from EP's src/config/smard.py.
"""

import re

# ── Short name registry ──────────────────────────────────────────────
# Used by the suffix DSL parser. Keys are the concise names that appear
# in feature strings. Values are the actual DataFrame column names.

SHORT_NAMES: dict[str, str] = {
    # Target
    "price": "target_price",
    # Generation (actuals)
    "gen_wind_on": "stromerzeugung_wind_onshore",
    "gen_wind_off": "stromerzeugung_wind_offshore",
    "gen_solar": "stromerzeugung_photovoltaik",
    "gen_nuclear": "stromerzeugung_kernenergie",
    "gen_lignite": "stromerzeugung_braunkohle",
    "gen_coal": "stromerzeugung_steinkohle",
    "gen_gas": "stromerzeugung_erdgas",
    "gen_hydro": "stromerzeugung_wasserkraft",
    "gen_biomass": "stromerzeugung_biomasse",
    "gen_pumped": "stromerzeugung_pumpspeicher",
    "gen_other": "stromerzeugung_sonstige_konventionelle",
    "gen_other_renew": "stromerzeugung_sonstige_erneuerbare",
    # Forecasts (published for today — no lag required)
    "prog_load": "prognostizierter_verbrauch_gesamt",
    "prog_gen_total": "prognostizierte_erzeugung_gesamt",
    "prog_gen_wind_pv": "prognostizierte_erzeugung_wind_und_photovoltaik",
    "prog_gen_wind_on": "prognostizierte_erzeugung_onshore",
    "prog_gen_wind_off": "prognostizierte_erzeugung_offshore",
    "prog_gen_solar": "prognostizierte_erzeugung_photovoltaik",
    "prog_gen_other": "prognostizierte_erzeugung_sonstige",
    "prog_residual": "prognostizierter_verbrauch_residuallast",
    # Consumption
    "load": "stromverbrauch_gesamt_(netzlast)",
    "residual_load": "stromverbrauch_residuallast",
    # Commodities
    "carbon": "carbon_eur_per_ton",
    "carbon_rt": "carbon_realtime_eur_per_ton",
    "ttf": "ttf_eur_per_mwh",
    "brent": "brent_usd_per_barrel",
    # Neighbour prices (for spread computation)
    "price_fr": "marktpreis_frankreich",
    "price_nl": "marktpreis_niederlande",
    "price_at": "marktpreis_oesterreich",
    "price_dk1": "marktpreis_daenemark_1",
    "price_dk2": "marktpreis_daenemark_2",
    "price_cz": "marktpreis_tschechien",
    "price_pl": "marktpreis_polen",
    "price_ch": "marktpreis_schweiz",
    "price_no2": "marktpreis_norwegen_2",
    "price_se4": "marktpreis_schweden_4",
    "price_be": "marktpreis_belgien",
    "price_hu": "marktpreis_ungarn",
    "price_si": "marktpreis_slowenien",
    "price_it_n": "marktpreis_italien_(nord)",
    # Derived (computed during feature engineering, not raw columns)
    "net_export_fr": "_derived_net_export_frankreich",
    "net_export_nl": "_derived_net_export_niederlande",
    "net_export_at": "_derived_net_export_oesterreich",
    "net_export_dk1": "_derived_net_export_daenemark_1",
    "net_export_dk2": "_derived_net_export_daenemark_2",
    "net_export_cz": "_derived_net_export_tschechien",
    "net_export_pl": "_derived_net_export_polen",
    "net_export_ch": "_derived_net_export_schweiz",
    "net_export_no2": "_derived_net_export_norwegen_2",
    "net_export_se4": "_derived_net_export_schweden_4",
    "net_export_be": "_derived_net_export_belgien",
    "net_export_hu": "_derived_net_export_ungarn",
    "net_export_si": "_derived_net_export_slowenien",
    "net_export_it_n": "_derived_net_export_italien_nord",
    "spread_fr": "_derived_spread_frankreich",
    "spread_nl": "_derived_spread_niederlande",
    "spread_at": "_derived_spread_oesterreich",
    "spread_dk1": "_derived_spread_daenemark_1",
    "spread_dk2": "_derived_spread_daenemark_2",
    "spread_cz": "_derived_spread_tschechien",
    "spread_pl": "_derived_spread_polen",
    "spread_ch": "_derived_spread_schweiz",
    "spread_no2": "_derived_spread_norwegen_2",
    "spread_se4": "_derived_spread_schweden_4",
    "spread_be": "_derived_spread_belgien",
    "spread_hu": "_derived_spread_ungarn",
    "spread_si": "_derived_spread_slowenien",
    "spread_it_n": "_derived_spread_italien_nord",
    "gen_pct_wind_on": "_derived_gen_pct_wind_onshore",
    "gen_pct_wind_off": "_derived_gen_pct_wind_offshore",
    "gen_pct_solar": "_derived_gen_pct_photovoltaik",
    "gen_pct_nuclear": "_derived_gen_pct_kernenergie",
    "gen_pct_lignite": "_derived_gen_pct_braunkohle",
    "gen_pct_coal": "_derived_gen_pct_steinkohle",
    "gen_pct_gas": "_derived_gen_pct_erdgas",
    "gen_pct_hydro": "_derived_gen_pct_wasserkraft",
    "gen_pct_biomass": "_derived_gen_pct_biomasse",
    "gen_pct_pumped": "_derived_gen_pct_pumpspeicher",
    "gen_pct_other": "_derived_gen_pct_sonstige_konventionelle",
    "gen_pct_other_renew": "_derived_gen_pct_sonstige_erneuerbare",
    # Temporal / deterministic (computed by engine, not raw columns)
    "hour": "_derived_hour_of_day",  # base for Fourier features
    "hour_of_day": "_derived_hour_of_day_int",  # raw integer 0-23
    "day_of_week": "_derived_day_of_week_int",  # raw integer 0-6
    "hour_sin": "_derived_hour_sin",
    "hour_cos": "_derived_hour_cos",
    "dow_sin": "_derived_dow_sin",
    "dow_cos": "_derived_dow_cos",
    "month_sin": "_derived_month_sin",
    "month_cos": "_derived_month_cos",
    "is_weekend": "_derived_is_weekend",
    "is_holiday": "_derived_is_holiday",
    "day_index": "_derived_day_index",
    "year_index": "_derived_year_index",
    # Derived market features (computed during feature engineering)
    "total_exports": "_derived_total_exports",
    "total_imports": "_derived_total_imports",
    "total_generation": "_derived_total_generation",
    "pct_renewable": "_derived_pct_renewable",
    "supply_demand_gap": "_derived_supply_demand_gap",
    "pct_prog_other": "_derived_pct_prog_other",
    "pct_prog_wind_pv": "_derived_pct_prog_wind_pv",
    "pct_prog_solar": "_derived_pct_prog_solar",
    "pct_prog_wind_on": "_derived_pct_prog_wind_on",
    "pct_prog_wind_off": "_derived_pct_prog_wind_off",
    # EEG regime indicator (deterministic, date-driven)
    "eeg_regime": "_derived_eeg_regime",
    # Negative-price rolling stats (computed on target price, lag via _d1)
    "neg_price_frac_30d": "_derived_neg_price_frac_30d",
    "neg_price_frac_90d": "_derived_neg_price_frac_90d",
    "neg_price_depth_30d": "_derived_neg_price_depth_30d",
    # EMA gen/load historical_forecasts are no longer exposed as separate
    # features — they overlay onto prog_* columns at dataset prep (waterfall
    # EMA → SMARD → actuals). See energy_forecasting.modeling.price._overlay_ema_forecasts.
    # Weather-derived (computed by weather FE classes, stage 4)
    "wpd_offshore_cap": "_derived_wpd_offshore_cap_weighted",
    "wpd_onshore_cap": "_derived_wpd_onshore_cap_weighted",
    "temp_cities_pop": "_derived_temperature_cities_pop_weighted",
    "ghi_solar_cap": "_derived_ghi_solar_cap_weighted",
}

REVERSE_SHORT_NAMES: dict[str, str] = {v: k for k, v in SHORT_NAMES.items()}


# ── SMARD filter key mappings ────────────────────────────────────────
# Ported from EP's src/config/smard.py. Maps SMARD API integer filter
# keys to German descriptions and snake_case column names.


def clean_column_name(description: str) -> str:
    """Convert German SMARD description to snake_case column name.

    >>> clean_column_name("Stromerzeugung: Braunkohle")
    'stromerzeugung_braunkohle'
    >>> clean_column_name("Marktpreis: Österreich")
    'marktpreis_oesterreich'
    """
    text = description.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(":", "_")
    # Transliterate German umlauts to ASCII
    text = text.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
    text = text.replace("Ö", "Oe").replace("Ä", "Ae").replace("Ü", "Ue")
    text = re.sub(r'[<>"|?*]', "", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text.lower()


# Full SMARD filter_dict (ported from EP's src/config/smard.py)
SMARD_FILTER_KEYS: dict[int, str] = {
    # Generation
    1223: "Stromerzeugung: Braunkohle",
    1224: "Stromerzeugung: Kernenergie",
    1225: "Stromerzeugung: Wind Offshore",
    1226: "Stromerzeugung: Wasserkraft",
    1227: "Stromerzeugung: Sonstige Konventionelle",
    1228: "Stromerzeugung: Sonstige Erneuerbare",
    4066: "Stromerzeugung: Biomasse",
    4067: "Stromerzeugung: Wind Onshore",
    4068: "Stromerzeugung: Photovoltaik",
    4069: "Stromerzeugung: Steinkohle",
    4070: "Stromerzeugung: Pumpspeicher",
    4071: "Stromerzeugung: Erdgas",
    # Forecasted generation
    3791: "Prognostizierte Erzeugung: Offshore",
    123: "Prognostizierte Erzeugung: Onshore",
    125: "Prognostizierte Erzeugung: Photovoltaik",
    715: "Prognostizierte Erzeugung: Sonstige",
    5097: "Prognostizierte Erzeugung: Wind und Photovoltaik",
    122: "Prognostizierte Erzeugung: Gesamt",
    # Demand
    410: "Stromverbrauch: Gesamt (Netzlast)",
    4359: "Stromverbrauch: Residuallast",
    4387: "Stromverbrauch: Pumpspeicher",
    # Forecasted demand
    411: "Prognostizierter Verbrauch: Gesamt",
    4362: "Prognostizierter Verbrauch: Residuallast",
    # Installed capacity
    186: "Installierte Erzeugungsleistung: Kernenergie",
    188: "Installierte Erzeugungsleistung: Braunkohle",
    189: "Installierte Erzeugungsleistung: Steinkohle",
    194: "Installierte Erzeugungsleistung: Erdgas",
    198: "Installierte Erzeugungsleistung: Pumpspeicher",
    207: "Installierte Erzeugungsleistung: Sonstige Konventionelle",
    3792: "Installierte Erzeugungsleistung: Wind Offshore",
    4072: "Installierte Erzeugungsleistung: Biomasse",
    4073: "Installierte Erzeugungsleistung: Wasserkraft",
    4074: "Installierte Erzeugungsleistung: Wind Onshore",
    4075: "Installierte Erzeugungsleistung: Photovoltaik",
    4076: "Installierte Erzeugungsleistung: Sonstige Erneuerbare",
    # Prices
    4169: "Marktpreis: Deutschland/Luxemburg",
    5078: "Marktpreis: Anrainer DE/LU",
    4996: "Marktpreis: Belgien",
    4997: "Marktpreis: Norwegen 2",
    4170: "Marktpreis: Österreich",
    251: "Marktpreis: Deutschland/Österreich/Luxemburg",
    252: "Marktpreis: Dänemark 1",
    253: "Marktpreis: Dänemark 2",
    254: "Marktpreis: Frankreich",
    255: "Marktpreis: Italien (Nord)",
    256: "Marktpreis: Niederlande",
    257: "Marktpreis: Polen",
    258: "Marktpreis: Schweden 4",
    259: "Marktpreis: Schweiz",
    260: "Marktpreis: Slowenien",
    261: "Marktpreis: Tschechien",
    262: "Marktpreis: Ungarn",
}

SMARD_COLUMN_NAMES: dict[int, str] = {
    k: clean_column_name(v) for k, v in SMARD_FILTER_KEYS.items()
}

# Cross-border physical flows (DE-LU bidding area, post-2018-09-30)
CROSS_BORDER_DE_LU: dict[int, str] = {
    4963: "Cross-Border Flows: net export",
    # Imports
    4840: "Cross-Border Flows: Denmark 1 imports",
    4841: "Cross-Border Flows: Denmark 2 imports",
    4842: "Cross-Border Flows: France imports",
    4843: "Cross-Border Flows: Netherlands imports",
    4844: "Cross-Border Flows: Poland imports",
    4845: "Cross-Border Flows: Sweden 4 imports",
    4846: "Cross-Border Flows: Switzerland imports",
    4847: "Cross-Border Flows: Czech Republic imports",
    4848: "Cross-Border Flows: Austria imports",
    4978: "Cross-Border Flows: Norway 2 imports",
    4982: "Cross-Border Flows: Belgium imports",
    # Exports
    4821: "Cross-Border Flows: Denmark 1 exports",
    4822: "Cross-Border Flows: Denmark 2 exports",
    4823: "Cross-Border Flows: France exports",
    4824: "Cross-Border Flows: Netherlands exports",
    4825: "Cross-Border Flows: Poland exports",
    4826: "Cross-Border Flows: Sweden 4 exports",
    4827: "Cross-Border Flows: Switzerland exports",
    4828: "Cross-Border Flows: Czech Republic exports",
    4829: "Cross-Border Flows: Austria exports",
    4976: "Cross-Border Flows: Norway 2 exports",
    4980: "Cross-Border Flows: Belgium exports",
}

# Cross-border physical flows (DE-AT-LU bidding area, pre-2018-09-30)
CROSS_BORDER_DE_AT_LU: dict[int, str] = {
    4963: "Cross-Border Flows: net export",
    # Denmark 1
    4872: "Cross-Border Flows: Denmark 1 exports",
    4726: "Cross-Border Flows: Denmark 1 imports",
    # Denmark 2
    4869: "Cross-Border Flows: Denmark 2 exports",
    4727: "Cross-Border Flows: Denmark 2 imports",
    # Netherlands
    4870: "Cross-Border Flows: Netherlands exports",
    4730: "Cross-Border Flows: Netherlands imports",
    # Northern Italy (Austria's neighbor — only in DE-AT-LU)
    4873: "Cross-Border Flows: Northern Italy exports",
    4729: "Cross-Border Flows: Northern Italy imports",
    # Switzerland
    4732: "Cross-Border Flows: Switzerland exports",
    4876: "Cross-Border Flows: Switzerland imports",
    # Czech Republic
    4734: "Cross-Border Flows: Czech Republic exports",
    4878: "Cross-Border Flows: Czech Republic imports",
    # France
    4728: "Cross-Border Flows: France exports",
    4871: "Cross-Border Flows: France imports",
    # Sweden 4
    4857: "Cross-Border Flows: Sweden 4 exports",
    4875: "Cross-Border Flows: Sweden 4 imports",
    # Hungary (Austria's neighbor — only in DE-AT-LU)
    4735: "Cross-Border Flows: Hungary exports",
    4879: "Cross-Border Flows: Hungary imports",
    # Slovenia (Austria's neighbor — only in DE-AT-LU)
    4733: "Cross-Border Flows: Slovenia exports",
    4877: "Cross-Border Flows: Slovenia imports",
    # Poland
    4731: "Cross-Border Flows: Poland exports",
    4874: "Cross-Border Flows: Poland imports",
    # Belgium (no data before ~2020)
    4984: "Cross-Border Flows: Belgium exports",
    4986: "Cross-Border Flows: Belgium imports",
}

# Installed capacity keys — excluded from data downloads
INSTALLED_CAPACITY_KEYS: set[int] = {
    186,
    188,
    189,
    194,
    198,
    207,
    3792,
    4072,
    4073,
    4074,
    4075,
    4076,
}

# Scheduled commercial exchange keys — excluded from data downloads
SCHEDULED_COMMERCIAL_KEYS: set[int] = {
    4546,
    4404,
    4548,
    4406,
    4712,
    4998,
    4553,
    4412,
    4552,
    4410,
    4550,
    4408,
    4724,
    4722,
    4545,
    4403,
    4551,
    4409,
    4547,
    4405,
    4549,
    4407,
    4629,
}

EXCLUDED_KEYS: set[int] = INSTALLED_CAPACITY_KEYS | SCHEDULED_COMMERCIAL_KEYS
