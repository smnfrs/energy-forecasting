"""Cleaning pipeline configuration.

Processing order matters: drop -> physical bounds -> structural fills
-> calculated fills -> correlate fills -> interpolation.

Ported from EP's handle_missing_values() and EMA's physical validation.
The config says *what* to clean and *why* (via comments); the helper
functions in data/processing.py say *how*.
"""

import pandas as pd

from energy_forecasting.data.processing import (
    clip_bounds,
    drop_columns,
    fill_from_column,
    fill_from_difference,
    fill_gen_total,
    fill_zero_after,
    fill_zero_before,
    fill_zero_before_first_valid,
    interpolate_gaps,
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules to the merged dataset.

    Each step has a comment documenting the domain rationale.
    Order matters — see module docstring.
    """

    # ── Drop redundant columns ──────────────────────────────────────
    # Values already captured in target_price
    df = drop_columns(
        df,
        [
            "marktpreis_deutschland_luxemburg",
            "marktpreis_deutschland_oesterreich_luxemburg",
            "marktpreis_anrainer_de_lu",
        ],
    )

    # ── Physical bounds (weather + market) ──────────────────────────
    # Ported from EMA's phys_limits; extended for market data
    df = clip_bounds(df, "temperature_2m_*", min_val=-45, max_val=50)
    df = clip_bounds(df, "wind_speed_*", min_val=0, max_val=200)
    df = clip_bounds(df, "relative_humidity_*", min_val=0, max_val=100)
    df = clip_bounds(df, "shortwave_radiation_*", min_val=0, max_val=1400)
    df = clip_bounds(df, "direct_radiation_*", min_val=0, max_val=1400)
    df = clip_bounds(df, "cloud_cover_*", min_val=0, max_val=100)
    df = clip_bounds(df, "target_price", min_val=-500, max_val=1000, action="nan")
    df = clip_bounds(df, "stromverbrauch_*", min_val=0)

    # ── Structural zero fills ───────────────────────────────────────
    # Nuclear: decommissioned April 2023
    df = fill_zero_after(df, "stromerzeugung_kernenergie", after="last_valid")

    # Austria neighbour flows: irrelevant after DE-AT-LU split
    df = fill_zero_after(
        df,
        [
            "cross-border_flows_hungary_exports",
            "cross-border_flows_hungary_imports",
            "cross-border_flows_slovenia_exports",
            "cross-border_flows_slovenia_imports",
        ],
        after="2018-09-30T22:00:00Z",
    )

    # Austria direct flows/price: didn't exist before split
    df = fill_zero_before(
        df,
        [
            "cross-border_flows_austria_exports",
            "cross-border_flows_austria_imports",
            "marktpreis_oesterreich",
        ],
        before="2018-09-30T22:00:00Z",
    )

    # Belgium flows: reporting started Oct 2017
    df = fill_zero_before(
        df,
        [
            "cross-border_flows_belgium_exports",
            "cross-border_flows_belgium_imports",
        ],
        before="2017-10-10T22:00:00Z",
    )

    # Norway flows: reporting started late
    df = fill_zero_before_first_valid(
        df,
        [
            "cross-border_flows_norway_2_exports",
            "cross-border_flows_norway_2_imports",
        ],
    )

    # ── Calculated fills ────────────────────────────────────────────
    # Other forecast = total - wind+PV forecast
    df = fill_from_difference(
        df,
        "prognostizierte_erzeugung_sonstige",
        total="prognostizierte_erzeugung_gesamt",
        subtract="prognostizierte_erzeugung_wind_und_photovoltaik",
    )

    # Load forecast backfilled from actual load (r=0.97)
    df = fill_from_column(
        df,
        "prognostizierter_verbrauch_gesamt",
        source="stromverbrauch_gesamt_(netzlast)",
    )

    # Generation total: complex logic (30-day recency check, component sum fallback)
    df = fill_gen_total(df)

    # Poland/Switzerland prices: zero-spread assumption
    df = fill_from_column(
        df,
        ["marktpreis_polen", "marktpreis_schweiz"],
        source="target_price",
    )

    # ── Final interpolation ─────────────────────────────────────────
    # Cubic spline for remaining small gaps (max 5 consecutive hours)
    df = interpolate_gaps(
        df,
        method="cubicspline",
        max_gap=5,
        exclude=["regime_de_at_lu", "regime_quarter_hourly", "target_price"],
    )

    return df
