"""Forecast output validation — fail fast before publishing bad data.

Raises ForecastValidationError (a subclass of ValueError) if any check fails.
All checks run before any JSON is written so a bad forecast never reaches the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ── EPEX Spot bounds (DE/AT/LU + safety buffer) ──────────────────────
PRICE_MIN = -500.0    # EUR/MWh
PRICE_MAX = 3_000.0   # EUR/MWh

# ── National generation / load plausibility bounds ───────────────────
LOAD_NATIONAL_MIN_MW = 10_000   # DE minimum total load
LOAD_NATIONAL_MAX_MW = 120_000  # DE peak load never exceeded
GENERATION_MIN_MW = 0.0          # generation cannot be negative
SOLAR_NIGHT_THRESHOLD_MW = 1.0   # solar output above this at night = suspect

# Night hours in local time where solar should be zero
SOLAR_NIGHT_HOURS = {0, 1, 2, 3, 4, 21, 22, 23}


class ForecastValidationError(ValueError):
    """Raised when a forecast output fails sanity checks."""


def _require_no_nan(df: pd.DataFrame, name: str, column: str = "y_pred") -> None:
    n = df[column].isna().sum()
    if n:
        raise ForecastValidationError(
            f"{name}: {n} NaN values in {column}"
        )


def _require_row_count(df: pd.DataFrame, expected: int, name: str) -> None:
    if len(df) != expected:
        raise ForecastValidationError(
            f"{name}: expected {expected} rows, got {len(df)}"
        )


def validate_price(price_df: pd.DataFrame) -> None:
    """Validate price forecast DataFrame.

    Expects 24 rows with columns [y_pred, y_lower, y_upper].
    """
    _require_row_count(price_df, 24, "price")
    _require_no_nan(price_df, "price", "y_pred")

    out_of_bounds = (
        (price_df["y_pred"] < PRICE_MIN) | (price_df["y_pred"] > PRICE_MAX)
    )
    if out_of_bounds.any():
        bad = price_df.loc[out_of_bounds, "y_pred"]
        raise ForecastValidationError(
            f"Price forecast has {out_of_bounds.sum()} values outside "
            f"[{PRICE_MIN}, {PRICE_MAX}] EUR/MWh: {bad.tolist()}"
        )
    logger.debug(
        f"Price validation OK: {len(price_df)} rows, "
        f"range [{price_df['y_pred'].min():.1f}, {price_df['y_pred'].max():.1f}]"
    )


def validate_generation(
    df: pd.DataFrame, name: str, *, is_solar: bool = False
) -> None:
    """Validate a gen/load forecast DataFrame.

    Expects 168 rows with column y_pred ≥ 0 for generation, with an extra
    check that solar is zero during night hours.
    """
    _require_row_count(df, 168, name)
    _require_no_nan(df, name, "y_pred")

    if (df["y_pred"] < GENERATION_MIN_MW).any():
        n_neg = (df["y_pred"] < GENERATION_MIN_MW).sum()
        raise ForecastValidationError(
            f"{name}: {n_neg} negative generation values "
            f"(min={df['y_pred'].min():.1f} MW)"
        )

    if is_solar:
        night_rows = df[df.index.hour.isin(SOLAR_NIGHT_HOURS)]
        night_positive = night_rows[night_rows["y_pred"] > SOLAR_NIGHT_THRESHOLD_MW]
        if not night_positive.empty:
            raise ForecastValidationError(
                f"{name} (solar): {len(night_positive)} night-hour rows have "
                f"positive output (>{SOLAR_NIGHT_THRESHOLD_MW} MW)"
            )


def validate_load(df: pd.DataFrame, name: str) -> None:
    """Validate a national load forecast DataFrame."""
    _require_row_count(df, 168, name)
    _require_no_nan(df, name, "y_pred")

    lo = df["y_pred"] < LOAD_NATIONAL_MIN_MW
    hi = df["y_pred"] > LOAD_NATIONAL_MAX_MW
    if lo.any() or hi.any():
        raise ForecastValidationError(
            f"{name}: {lo.sum()} rows below {LOAD_NATIONAL_MIN_MW} MW, "
            f"{hi.sum()} rows above {LOAD_NATIONAL_MAX_MW} MW"
        )


def validate_outputs(
    price_df: pd.DataFrame,
    gen_load_results: dict,
) -> None:
    """Run all validation checks. Raises ForecastValidationError on first failure.

    Parameters
    ----------
    price_df : 24-row price forecast
    gen_load_results : dict keyed by (target, region) → DataFrame
    """
    errors: list[str] = []

    def _check(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except ForecastValidationError as exc:
            errors.append(str(exc))

    # Price
    _check(validate_price, price_df)

    # Gen/load national (most reliable for plausibility bounds)
    for target in ["wind_onshore", "wind_offshore", "solar", "load"]:
        key = (target, "DE_NATIONAL")
        if key not in gen_load_results:
            errors.append(f"Missing national forecast for {target}")
            continue
        df = gen_load_results[key]
        if target == "solar":
            _check(validate_generation, df, f"{target}/DE_NATIONAL", is_solar=True)
        elif target == "load":
            _check(validate_load, df, "load/DE_NATIONAL")
        else:
            _check(validate_generation, df, f"{target}/DE_NATIONAL")

    if errors:
        msg = f"Forecast validation failed ({len(errors)} error(s)):\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ForecastValidationError(msg)

    logger.info("Forecast validation passed")
