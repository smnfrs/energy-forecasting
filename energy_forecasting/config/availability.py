"""Availability rules: when is each data source physically available for prediction?

Used by leakage validation (stage 4) to check that feature suffixes imply safe lags.
"""

from dataclasses import dataclass


@dataclass
class AvailabilityRule:
    """When is this column's data available for prediction?

    max_offset_days: how many days back is the latest available data?
        0 = today's data is available (forecasts, deterministic features)
       -1 = yesterday's data is the latest available
       -2 = two days ago (business day delay)
    cutoff_hour: hour (UTC) by which the data is published on the offset day.
        None = available all day (e.g., auction results published once).
    """

    pattern: str
    max_offset_days: int
    cutoff_hour: int | None
    reason: str


AVAILABILITY_RULES: list[AvailabilityRule] = [
    # Source-neutral own forecast inputs built before the price run.
    AvailabilityRule(
        "forecast_*",
        0,
        None,
        "Own gen/load forecast artifacts are produced before the 08:00 UTC price run.",
    ),
    AvailabilityRule("hour_*", 0, None, "Deterministic temporal features"),
    AvailabilityRule("dow_*", 0, None, "Deterministic temporal features"),
    AvailabilityRule("is_holiday", 0, None, "Deterministic"),
    AvailabilityRule("day_index", 0, None, "Deterministic"),
    AvailabilityRule("year_index", 0, None, "Deterministic"),
    AvailabilityRule("is_weekend", 0, None, "Deterministic"),
    AvailabilityRule("month_*", 0, None, "Deterministic temporal features"),
    AvailabilityRule("hour", 0, None, "Deterministic (Fourier base)"),
    AvailabilityRule("hour_of_day", 0, None, "Deterministic"),
    AvailabilityRule("day_of_week", 0, None, "Deterministic"),
    AvailabilityRule(
        "pct_forecast_*",
        0,
        None,
        "Derived from source-neutral forecast_* columns, same availability.",
    ),
    AvailabilityRule(
        "eeg_regime",
        0,
        None,
        "Deterministic regime indicator from date (EEG threshold dates).",
    ),
    # Weather forecasts — available for today (Open-Meteo forecast endpoint)
    AvailabilityRule("wpd_*", 0, None, "Weather forecasts available for delivery day"),
    AvailabilityRule("temp_*", 0, None, "Weather forecasts available for delivery day"),
    AvailabilityRule("ghi_*", 0, None, "Weather forecasts available for delivery day"),
    # Negative-price rolling stats — must precede the general price_* rule.
    # Computed on the target price column at dataset prep; the engine adds
    # the D-1 lag via the `_d1` suffix on the SHORT_NAME alias.
    AvailabilityRule(
        "neg_price_*",
        -1,
        None,
        "Rolling stats on target price; engine applies D-1 lag via _d1 suffix.",
    ),
    # Price — D-1 auction results published previous afternoon
    AvailabilityRule("price", -1, None, "EPEX SPOT auction results published D-1 ~13:00 CET"),
    AvailabilityRule("price_*", -1, None, "Neighbour prices published D-1"),
    AvailabilityRule("spread_*", -1, None, "Derived from prices, same availability"),
    # Generation/load actuals — SMARD data must be available 1h before 08:00 UTC inference.
    # Inference runs at 08:00 UTC (09:00 CET winter / 10:00 CEST summer); D-1 morning data
    # up to 07:00 UTC (08:00 CET winter) is the latest we can count on. `cutoff_hour=7`
    # means the validator requires `_eh7` or earlier for single-day D-1 features.
    AvailabilityRule("gen_*", -1, 7, "SMARD generation actuals — usable up to 07:00 UTC on D-1"),
    AvailabilityRule("load", -1, 7, "SMARD load actuals same delay as generation"),
    AvailabilityRule("net_export_*", -1, 7, "Derived from cross-border flows, same delay"),
    AvailabilityRule("gen_pct_*", -1, 7, "Derived from generation, same delay"),
    AvailabilityRule("residual_load", -1, 7, "SMARD residual load, same delay as generation"),
    AvailabilityRule("pct_renewable", -1, 7, "Derived from generation, same delay"),
    AvailabilityRule("supply_demand_gap", -1, 7, "Derived from generation + load, same delay"),
    AvailabilityRule("total_exports", -1, 7, "Derived from cross-border flows, same delay"),
    AvailabilityRule("total_imports", -1, 7, "Derived from cross-border flows, same delay"),
    AvailabilityRule("total_generation", -1, 7, "Derived from generation, same delay"),
    # Commodities — business day delay
    AvailabilityRule("carbon", -2, None, "ICAP carbon published with ~2 day lag"),
    AvailabilityRule("carbon_rt", -1, None, "CO2.L equity proxy, previous close"),
    AvailabilityRule("ttf", -2, None, "TTF futures, business day delay"),
    AvailabilityRule("brent", -2, None, "Brent futures, business day delay"),
]
