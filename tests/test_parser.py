"""Tests for features/parser.py — suffix DSL parser."""

import pytest
from energy_forecasting.features.parser import (
    EWMA,
    Aggregation,
    DailyAggregate,
    FeatureSpec,
    Fourier,
    HourlyLag,
    InteractionSpec,
    parse_feature,
)

# ── Bare short names (no suffix) ─────────────────────────────────


def test_bare_short_name():
    spec = parse_feature("price")
    assert isinstance(spec, FeatureSpec)
    assert spec.base == "price"
    assert spec.raw_col == "target_price"
    assert spec.lag is None
    assert spec.agg is None


def test_bare_temporal_name():
    spec = parse_feature("hour_sin")
    assert spec.base == "hour_sin"
    assert spec.raw_col == "_derived_hour_sin"


def test_bare_derived_name():
    spec = parse_feature("is_weekend")
    assert spec.base == "is_weekend"
    assert spec.raw_col == "_derived_is_weekend"


# ── Hourly lag ────────────────────────────────────────────────────


def test_hourly_lag_24():
    spec = parse_feature("price_h24")
    assert spec.lag == HourlyLag(hours=24)
    assert spec.agg is None


def test_hourly_lag_168():
    spec = parse_feature("price_h168")
    assert spec.lag == HourlyLag(hours=168)


def test_hourly_lag_1():
    spec = parse_feature("gen_wind_on_h1")
    assert spec.lag == HourlyLag(hours=1)
    assert spec.base == "gen_wind_on"


# ── Single-day aggregation (_dN shorthand) ────────────────────────


def test_single_day_agg():
    spec = parse_feature("price_d1")
    assert spec.agg == Aggregation(start_day=-1, end_day=-1, stat="avg")


def test_single_day_agg_commodity():
    spec = parse_feature("ttf_d2")
    assert spec.agg == Aggregation(start_day=-2, end_day=-2, stat="avg")


# ── Multi-day aggregation (_dX_dY) ────────────────────────────────


def test_multi_day_agg():
    spec = parse_feature("price_d7_d1")
    assert spec.agg == Aggregation(start_day=-7, end_day=-1, stat="avg")


def test_multi_day_agg_with_stat():
    spec = parse_feature("price_d7_d1_std")
    assert spec.agg == Aggregation(start_day=-7, end_day=-1, stat="std")


def test_multi_day_all_stats():
    for stat in ["avg", "std", "min", "max", "sum", "range"]:
        spec = parse_feature(f"price_d7_d1_{stat}")
        assert spec.agg.stat == stat


# ── End-hour (_eh) ────────────────────────────────────────────────


def test_end_hour():
    spec = parse_feature("gen_wind_on_d1_eh10")
    assert spec.agg == Aggregation(start_day=-1, end_day=-1, stat="avg", end_hour=10)


def test_end_hour_multi_day():
    spec = parse_feature("price_d7_d1_eh8")
    assert spec.agg == Aggregation(start_day=-7, end_day=-1, stat="avg", end_hour=8)


def test_end_hour_with_stat():
    spec = parse_feature("price_d7_d1_eh8_std")
    assert spec.agg == Aggregation(start_day=-7, end_day=-1, stat="std", end_hour=8)


# ── Hour filter (_hA_hB) ─────────────────────────────────────────


def test_hour_filter():
    spec = parse_feature("price_d7_d1_h8_h20_avg")
    assert spec.agg.hour_start == 8
    assert spec.agg.hour_end == 20
    assert spec.agg.stat == "avg"


def test_hour_filter_default_stat():
    spec = parse_feature("price_d7_d1_h8_h20")
    # hour filter parses, but trailing _h20 might be ambiguous
    # Let's check what actually happens
    assert spec.agg is not None


# ── End-hour vs hour filter mutual exclusivity ────────────────────


def test_eh_and_hh_mutually_exclusive():
    with pytest.raises(ValueError, match="Cannot combine _eh and _h_h"):
        parse_feature("price_d7_d1_eh8_h8_h20")


# ── EWMA ──────────────────────────────────────────────────────────


def test_ewma_basic():
    spec = parse_feature("price_ewma_6")
    assert spec.ewma == EWMA(span=6)


def test_ewma_with_cutoff_day():
    spec = parse_feature("price_ewma_6_d1")
    assert spec.ewma == EWMA(span=6, cutoff_day=-1)


def test_ewma_with_cutoff_day_and_hour():
    spec = parse_feature("price_ewma_6_d1_h10")
    assert spec.ewma == EWMA(span=6, cutoff_day=-1, cutoff_hour=10)


def test_ewma_large_span():
    spec = parse_feature("ttf_ewma_720_d2")
    assert spec.ewma == EWMA(span=720, cutoff_day=-2)


# ── Fourier ───────────────────────────────────────────────────────


def test_fourier():
    spec = parse_feature("hour_fourier_24_3")
    assert spec.fourier == Fourier(period=24, order=3)
    assert spec.base == "hour"


def test_fourier_weekly():
    spec = parse_feature("hour_fourier_168_2")
    assert spec.fourier == Fourier(period=168, order=2)


# ── Daily aggregate ───────────────────────────────────────────────


def test_daily_agg():
    spec = parse_feature("forecast_gen_total_daily_sum")
    assert spec.daily_agg == DailyAggregate(stat="sum")


def test_daily_agg_max():
    spec = parse_feature("forecast_gen_solar_daily_max")
    assert spec.daily_agg == DailyAggregate(stat="max")


# ── Interactions ──────────────────────────────────────────────────


def test_interaction():
    spec = parse_feature("gen_wind_on_d1_eh10__x__day_index")
    assert isinstance(spec, InteractionSpec)
    assert spec.left.base == "gen_wind_on"
    assert spec.left.agg.end_hour == 10
    assert spec.right.base == "day_index"


def test_interaction_both_lagged():
    spec = parse_feature("price_d7_d1_std__x__is_weekend")
    assert isinstance(spec, InteractionSpec)
    assert spec.left.agg.stat == "std"
    assert spec.right.base == "is_weekend"


def test_nested_interaction_rejected():
    with pytest.raises(ValueError, match="Invalid interaction format"):
        parse_feature("a__x__b__x__c")


# ── Error cases ───────────────────────────────────────────────────


def test_unknown_short_name():
    with pytest.raises(ValueError, match="Unknown short name"):
        parse_feature("zzz_nonexistent_h24")


def test_invalid_suffix():
    with pytest.raises(ValueError, match="Invalid suffix"):
        parse_feature("price_xyz")


# ── Short name resolution (longest match) ─────────────────────────


def test_gen_wind_on_not_gen_wind():
    """gen_wind_on should match as short name, not gen_wind + _on suffix."""
    spec = parse_feature("gen_wind_on_h24")
    assert spec.base == "gen_wind_on"
    assert spec.raw_col == "stromerzeugung_wind_onshore"


def test_hour_sin_vs_hour():
    """hour_sin should match as 'hour_sin', not 'hour' + '_sin' suffix."""
    spec = parse_feature("hour_sin")
    assert spec.base == "hour_sin"

    # But hour_fourier should match 'hour' + '_fourier_...' suffix
    spec2 = parse_feature("hour_fourier_24_3")
    assert spec2.base == "hour"


def test_forecast_gen_total_resolution():
    spec = parse_feature("forecast_gen_total_daily_sum")
    assert spec.base == "forecast_gen_total"
    assert spec.raw_col == "forecast_gen_total"
