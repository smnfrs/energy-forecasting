"""Tests for deploy/validation.py."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.deploy.validation import (
    ForecastValidationError,
    validate_generation,
    validate_load,
    validate_outputs,
    validate_price,
)


def _price_df(values=None, n=24) -> pd.DataFrame:
    if values is None:
        values = np.ones(n) * 100.0
    idx = pd.date_range("2026-06-30 00:00", periods=n, freq="h")
    return pd.DataFrame(
        {"y_pred": values, "y_lower": values - 10, "y_upper": values + 10}, index=idx
    )


def _gen_df(values=None, n=168, start="2026-06-30 00:00") -> pd.DataFrame:
    if values is None:
        values = np.ones(n) * 5000.0
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {"y_pred": values, "y_lower": values * 0.9, "y_upper": values * 1.1}, index=idx
    )


def _load_df(values=None, n=168) -> pd.DataFrame:
    if values is None:
        values = np.ones(n) * 55_000.0
    idx = pd.date_range("2026-06-30 00:00", periods=n, freq="h")
    return pd.DataFrame(
        {"y_pred": values, "y_lower": values * 0.95, "y_upper": values * 1.05}, index=idx
    )


# ── Price ────────────────────────────────────────────────────────────


def test_price_valid():
    validate_price(_price_df())


def test_price_nan_raises():
    df = _price_df()
    df.iloc[5, df.columns.get_loc("y_pred")] = np.nan
    with pytest.raises(ForecastValidationError, match="NaN"):
        validate_price(df)


def test_price_too_low_raises():
    df = _price_df(np.full(24, -600.0))
    with pytest.raises(ForecastValidationError, match="-500"):
        validate_price(df)


def test_price_too_high_raises():
    df = _price_df(np.full(24, 5000.0))
    with pytest.raises(ForecastValidationError, match="3000"):
        validate_price(df)


def test_price_wrong_row_count():
    with pytest.raises(ForecastValidationError, match="expected 24"):
        validate_price(_price_df(n=23))


# ── Generation ───────────────────────────────────────────────────────


def test_generation_valid():
    validate_generation(_gen_df(), "wind_onshore/DE_NATIONAL")


def test_generation_negative_raises():
    df = _gen_df(np.full(168, -100.0))
    with pytest.raises(ForecastValidationError, match="negative"):
        validate_generation(df, "wind_onshore/DE_NATIONAL")


def test_solar_night_positive_raises():
    # Midnight hour (hour=0) with non-zero solar
    df = _gen_df(values=np.ones(168) * 0.0)  # all zero
    # Set hour 0 of day 1 to high value
    df.iloc[0, df.columns.get_loc("y_pred")] = 100.0  # hour 0 = midnight
    with pytest.raises(ForecastValidationError, match="night"):
        validate_generation(df, "solar/DE_NATIONAL", is_solar=True)


def test_solar_day_positive_ok():
    # Only daytime hours positive — should pass
    df = _gen_df(values=np.zeros(168))
    # Set daytime hours to 5000 MW
    for i in range(168):
        hour = df.index[i].hour
        if 5 <= hour <= 19:
            df.iloc[i, df.columns.get_loc("y_pred")] = 5000.0
    validate_generation(df, "solar/DE_NATIONAL", is_solar=True)


# ── Load ─────────────────────────────────────────────────────────────


def test_load_valid():
    validate_load(_load_df(), "load/DE_NATIONAL")


def test_load_too_low():
    df = _load_df(np.full(168, 5_000.0))
    with pytest.raises(ForecastValidationError, match="10000"):
        validate_load(df, "load/DE_NATIONAL")


def test_load_too_high():
    df = _load_df(np.full(168, 150_000.0))
    with pytest.raises(ForecastValidationError, match="120000"):
        validate_load(df, "load/DE_NATIONAL")


# ── validate_outputs ─────────────────────────────────────────────────


def test_validate_outputs_clean():
    price = _price_df()
    gen_load = {
        ("wind_onshore", "DE_NATIONAL"): _gen_df(),
        ("wind_offshore", "DE_NATIONAL"): _gen_df(),
        ("solar", "DE_NATIONAL"): _gen_df(values=np.zeros(168)),
        ("load", "DE_NATIONAL"): _load_df(),
    }
    validate_outputs(price, gen_load)


def test_validate_outputs_collects_all_errors():
    price = _price_df(np.full(24, 5000.0))  # too high
    gen_load = {
        ("wind_onshore", "DE_NATIONAL"): _gen_df(np.full(168, -1.0)),  # negative
        ("wind_offshore", "DE_NATIONAL"): _gen_df(),
        ("solar", "DE_NATIONAL"): _gen_df(values=np.zeros(168)),
        ("load", "DE_NATIONAL"): _load_df(),
    }
    with pytest.raises(ForecastValidationError) as exc_info:
        validate_outputs(price, gen_load)
    # Both errors should appear
    assert "2 error" in str(exc_info.value)
