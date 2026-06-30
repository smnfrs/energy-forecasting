"""Tests for data/merge.py merge pipeline functions."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.config.merge import BIDDING_AREA_SPLIT, QUARTER_HOURLY_START
from energy_forecasting.data.merge import (
    add_regime_indicators,
    create_unified_target,
    enforce_periodicity,
    extend_with_energy_charts,
    impute_medium_gaps,
    merge_commodities,
    merge_national_smard,
    normalize_dst,
    validate_no_nans,
    warn_physical_bounds,
)
from loguru import logger

# -- Fixtures ---------------------------------------------------------------


@pytest.fixture
def hourly_df():
    """Complete hourly DataFrame with no gaps."""
    idx = pd.date_range("2020-01-01", periods=72, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "col_a": rng.normal(100, 10, 72),
            "col_b": rng.normal(50, 5, 72),
        },
        index=idx,
    )


# -- enforce_periodicity tests ----------------------------------------------


def test_enforce_periodicity_no_gaps(hourly_df):
    result = enforce_periodicity(hourly_df)
    pd.testing.assert_frame_equal(result, hourly_df)


def test_enforce_periodicity_fills_small_gap():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"x": range(10)}, index=idx, dtype=float)
    # Remove 2 hours to create a gap
    df = df.drop(idx[4:6])
    assert len(df) == 8

    result = enforce_periodicity(df, max_gap=3)
    assert len(result) == 10
    assert not result["x"].isna().any()


def test_enforce_periodicity_rejects_large_gap():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"x": range(10)}, index=idx, dtype=float)
    # Remove 4 hours -> exceeds max_gap=3
    df = df.drop(idx[3:7])

    with pytest.raises(ValueError, match="exceeds max_gap"):
        enforce_periodicity(df, max_gap=3)


# -- impute_medium_gaps tests -----------------------------------------------


def test_impute_medium_gaps_fills_day_gap():
    # Create 30 days of hourly data with a 24h gap in the middle
    idx = pd.date_range("2020-01-01", periods=720, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    # Diurnal pattern: values depend on hour
    values = np.array([50.0 + 10 * np.sin(2 * np.pi * h / 24) for h in range(720)])
    values += rng.normal(0, 1, 720)
    df = pd.DataFrame({"x": values}, index=idx)

    # Create a 24h gap (medium: 6 < 24 <= 48)
    gap_start = 360  # day 15
    df.iloc[gap_start : gap_start + 24, 0] = np.nan

    result = impute_medium_gaps(df, small_gap_max=5, medium_gap_max=48, window_days=14)

    # Gap should be filled
    assert not result.iloc[gap_start : gap_start + 24]["x"].isna().any()

    # Filled values should roughly match the diurnal pattern
    for i in range(24):
        filled_val = result.iloc[gap_start + i]["x"]
        hour = idx[gap_start + i].hour
        expected_approx = 50.0 + 10 * np.sin(2 * np.pi * hour / 24)
        assert abs(filled_val - expected_approx) < 10, (
            f"Hour {hour}: filled={filled_val:.1f}, expected~{expected_approx:.1f}"
        )


def test_impute_medium_gaps_skips_small_gaps():
    idx = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    values = np.arange(100, dtype=float)
    values[10:13] = np.nan  # 3h gap <= small_gap_max=5
    df = pd.DataFrame({"x": values}, index=idx)

    result = impute_medium_gaps(df, small_gap_max=5, medium_gap_max=48)
    # Small gap should NOT be filled (left for clean())
    assert result.iloc[10:13]["x"].isna().all()


def test_impute_medium_gaps_skips_large_gaps():
    idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    values = np.arange(200, dtype=float)
    values[50:110] = np.nan  # 60h gap > medium_gap_max=48
    df = pd.DataFrame({"x": values}, index=idx)

    result = impute_medium_gaps(df, small_gap_max=5, medium_gap_max=48)
    # Large gap should NOT be filled
    assert result.iloc[50:110]["x"].isna().all()


def test_impute_medium_gaps_excludes_columns():
    idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    values = np.arange(200, dtype=float)
    gap_vals = values.copy()
    gap_vals[50:70] = np.nan  # 20h gap (medium)
    df = pd.DataFrame({"x": gap_vals, "regime": gap_vals.copy()}, index=idx)

    result = impute_medium_gaps(df, small_gap_max=5, medium_gap_max=48, exclude=["regime"])
    # x should be filled, regime should not
    assert not result.iloc[50:70]["x"].isna().any()
    assert result.iloc[50:70]["regime"].isna().all()


# -- merge_national_smard tests ---------------------------------------------


def test_merge_national_smard_splits_at_cutoff():
    cutoff = pd.Timestamp("2020-06-01", tz="UTC")
    idx_pre = pd.date_range("2020-01-01", "2020-07-01", freq="h", tz="UTC")
    idx_post = pd.date_range("2020-05-01", "2020-12-01", freq="h", tz="UTC")

    df_pre = pd.DataFrame({"pre_col": 1.0}, index=idx_pre)
    df_post = pd.DataFrame({"post_col": 2.0}, index=idx_post)

    result = merge_national_smard(df_post, df_pre, cutoff=cutoff)

    # Before cutoff: should come from pre
    pre_slice = result[result.index < cutoff]
    assert pre_slice["pre_col"].notna().all()

    # From cutoff onwards: should come from post
    post_slice = result[result.index >= cutoff]
    assert post_slice["post_col"].notna().all()


def test_merge_national_smard_outer_columns():
    cutoff = pd.Timestamp("2020-06-01", tz="UTC")
    idx_pre = pd.date_range("2020-01-01", "2020-05-31", freq="D", tz="UTC")
    idx_post = pd.date_range("2020-06-01", "2020-12-31", freq="D", tz="UTC")

    df_pre = pd.DataFrame({"shared": 1.0, "only_pre": 10.0}, index=idx_pre)
    df_post = pd.DataFrame({"shared": 2.0, "only_post": 20.0}, index=idx_post)

    result = merge_national_smard(df_post, df_pre, cutoff=cutoff)

    # Both unique columns should exist
    assert "only_pre" in result.columns
    assert "only_post" in result.columns

    # only_pre should be NaN in post period
    post_slice = result[result.index >= cutoff]
    assert post_slice["only_pre"].isna().all()


# -- create_unified_target tests --------------------------------------------


def test_create_unified_target_post_priority():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "marktpreis_deutschland_luxemburg": [10, 20, 30, 40, 50],
            "marktpreis_deutschland_oesterreich_luxemburg": [1, 2, 3, 4, 5],
        },
        index=idx,
        dtype=float,
    )
    result = create_unified_target(df)
    # Post-split should take priority
    assert list(result["target_price"]) == [10, 20, 30, 40, 50]


def test_create_unified_target_falls_back_to_pre():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "marktpreis_deutschland_luxemburg": [np.nan, 20, np.nan, 40, np.nan],
            "marktpreis_deutschland_oesterreich_luxemburg": [1, 2, 3, 4, 5],
        },
        index=idx,
        dtype=float,
    )
    result = create_unified_target(df)
    assert result.loc[idx[0], "target_price"] == 1  # fallback to pre
    assert result.loc[idx[1], "target_price"] == 20  # post available


def test_create_unified_target_ec_fallback():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "marktpreis_deutschland_luxemburg": [np.nan, 20, np.nan, 40, np.nan],
            "marktpreis_deutschland_oesterreich_luxemburg": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        },
        index=idx,
        dtype=float,
    )
    ec = pd.Series([100, 200, 300, 400, 500], index=idx, dtype=float)
    result = create_unified_target(df, ec_fallback=ec)
    assert result.loc[idx[0], "target_price"] == 100  # EC fallback
    assert result.loc[idx[1], "target_price"] == 20  # post available


# -- extend_with_energy_charts tests ----------------------------------------


def test_extend_appends_beyond_smard():
    smard_idx = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
    ec_idx = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC")

    df = pd.DataFrame({"target_price": 50.0, "col_a": 1.0}, index=smard_idx)
    ec = pd.Series(99.0, index=ec_idx)

    result = extend_with_energy_charts(df, ec)
    assert len(result) == 48
    # New rows should have target_price from EC
    assert result.loc[ec_idx[-1], "target_price"] == 99.0


def test_extend_resamples_quarter_hourly():
    smard_idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    # Quarter-hourly EC data beyond SMARD
    ec_idx = pd.date_range("2020-01-01T02:00", periods=8, freq="15min", tz="UTC")
    ec = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=ec_idx, dtype=float)

    df = pd.DataFrame({"target_price": 50.0}, index=smard_idx)
    result = extend_with_energy_charts(df, ec)

    # 8 quarter-hourly = 2 hourly, so total = 2 + 2 = 4
    assert len(result) == 4
    # First EC hour should be mean(10, 20, 30, 40) = 25
    assert result.loc[pd.Timestamp("2020-01-01T02:00", tz="UTC"), "target_price"] == 25.0


def test_extend_noop_when_no_ec():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"target_price": 50.0}, index=idx)
    result = extend_with_energy_charts(df, None)
    assert len(result) == 10


# -- add_regime_indicators tests --------------------------------------------


def test_add_regime_indicators_de_at_lu():
    idx = pd.DatetimeIndex(
        [
            BIDDING_AREA_SPLIT - pd.Timedelta(hours=1),
            BIDDING_AREA_SPLIT,
            BIDDING_AREA_SPLIT + pd.Timedelta(hours=1),
        ]
    )
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    result = add_regime_indicators(df)
    assert result.loc[idx[0], "regime_de_at_lu"] == 1
    assert result.loc[idx[1], "regime_de_at_lu"] == 0
    assert result.loc[idx[2], "regime_de_at_lu"] == 0


def test_add_regime_indicators_quarter_hourly():
    idx = pd.DatetimeIndex(
        [
            QUARTER_HOURLY_START - pd.Timedelta(hours=1),
            QUARTER_HOURLY_START,
            QUARTER_HOURLY_START + pd.Timedelta(hours=1),
        ]
    )
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    result = add_regime_indicators(df)
    assert result.loc[idx[0], "regime_quarter_hourly"] == 0
    assert result.loc[idx[1], "regime_quarter_hourly"] == 1
    assert result.loc[idx[2], "regime_quarter_hourly"] == 1


# -- merge_commodities tests ------------------------------------------------


def test_merge_commodities_forward_fills():
    hourly_idx = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC")
    daily_idx = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")

    df = pd.DataFrame({"x": 1.0}, index=hourly_idx)
    commodity = pd.DataFrame({"ttf": [10.0, 20.0]}, index=daily_idx)

    result = merge_commodities(df, commodity)
    # First 24 hours should get day 1 price
    assert result.loc[hourly_idx[0], "ttf"] == 10.0
    assert result.loc[hourly_idx[23], "ttf"] == 10.0
    # Next 24 hours should get day 2 price
    assert result.loc[hourly_idx[24], "ttf"] == 20.0


def test_merge_commodities_aligns_utc_midnight():
    # Hours that span midnight should map to the correct day
    hourly_idx = pd.date_range("2020-01-01 20:00", periods=10, freq="h", tz="UTC")
    daily_idx = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")

    df = pd.DataFrame({"x": 1.0}, index=hourly_idx)
    commodity = pd.DataFrame({"ttf": [10.0, 20.0]}, index=daily_idx)

    result = merge_commodities(df, commodity)
    # 2020-01-01 20:00 UTC normalizes to 2020-01-01 -> day 1 price
    assert result.loc[hourly_idx[0], "ttf"] == 10.0
    # 2020-01-02 02:00 UTC normalizes to 2020-01-02 -> day 2 price
    assert result.loc[hourly_idx[6], "ttf"] == 20.0


# -- normalize_dst tests ----------------------------------------------------


def test_normalize_dst_spring_forward():
    # March 29, 2020: CET->CEST. Local day runs from
    # UTC 2020-03-28 23:00 (=00:00 CET) to 2020-03-29 21:00 (=23:00 CEST).
    # That's 23 UTC hours -> 23 local hours (missing local hour 2).
    idx = pd.date_range("2020-03-28 23:00", periods=23, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(23, dtype=float)}, index=idx)

    result = normalize_dst(df, timezone="Europe/Berlin")
    from datetime import date

    day_data = result[result.index.date == date(2020, 3, 29)]
    assert len(day_data) == 24
    # The interpolated hour 2 should exist
    assert 2 in set(day_data.index.hour)


def test_normalize_dst_fall_back():
    # October 25, 2020: CEST->CET. Local day runs from
    # UTC 2020-10-24 22:00 (=00:00 CEST) to 2020-10-25 22:00 (=23:00 CET).
    # That's 25 UTC hours -> 25 local hours (hour 2 appears twice).
    idx = pd.date_range("2020-10-24 22:00", periods=25, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(25, dtype=float)}, index=idx)

    result = normalize_dst(df, timezone="Europe/Berlin")
    from datetime import date

    day_data = result[result.index.date == date(2020, 10, 25)]
    assert len(day_data) == 24
    # No duplicate hours
    assert len(set(day_data.index.hour)) == 24


def test_normalize_dst_output_is_naive_local():
    idx = pd.date_range("2020-06-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"x": range(24)}, index=idx, dtype=float)
    result = normalize_dst(df)
    # Output is tz-naive (representing Europe/Berlin delivery hours)
    assert result.index.tz is None
    # Hours should be offset from UTC (summer: CEST = UTC+2)
    assert result.index[0].hour == 2  # 00:00 UTC = 02:00 CEST


def test_normalize_dst_normal_day():
    idx = pd.date_range("2020-06-15", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"x": np.arange(24, dtype=float)}, index=idx)
    result = normalize_dst(df)
    assert len(result) == 24


# -- validate_no_nans tests -------------------------------------------------


def test_validate_no_nans_warns_critical(caplog):
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"target_price": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]}, index=idx)

    # Use loguru's sink to capture
    messages = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        validate_no_nans(df, critical_cols=["target_price"])
    finally:
        logger.remove(handler_id)

    assert any("CRITICAL" in m and "target_price" in m for m in messages)


def test_validate_no_nans_passes_clean():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"target_price": range(10), "col_a": range(10)}, index=idx)

    messages = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        validate_no_nans(df)
    finally:
        logger.remove(handler_id)

    assert not any("CRITICAL" in m for m in messages)


# -- warn_physical_bounds tests ---------------------------------------------


def test_warn_physical_bounds_logs_violations():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "stromerzeugung_wind_onshore": [
                -100,  # below 0
                50000,
                50000,
                50000,
                50000,
                50000,
                50000,
                50000,
                50000,
                200_000,  # above 100k
            ]
        },
        index=idx,
        dtype=float,
    )

    messages = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        warn_physical_bounds(df)
    finally:
        logger.remove(handler_id)

    assert any("stromerzeugung_wind_onshore" in m for m in messages)


# -- Smoke test for run_merge_pipeline --------------------------------------


def test_run_merge_pipeline_smoke(tmp_path):
    """End-to-end smoke test with synthetic data."""
    from energy_forecasting.data.io import save_parquet

    smard_dir = tmp_path / "smard"
    smard_dir.mkdir()

    # Create synthetic DE-LU data (post-split only for simplicity)
    idx_post = pd.date_range("2019-01-01", "2020-01-01", freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    de_lu = pd.DataFrame(
        {
            "marktpreis_deutschland_luxemburg": rng.normal(50, 10, len(idx_post)),
            "stromerzeugung_wind_onshore": rng.uniform(1000, 20000, len(idx_post)),
            "stromerzeugung_photovoltaik": rng.uniform(0, 15000, len(idx_post)),
            "stromverbrauch_gesamt_(netzlast)": rng.uniform(40000, 80000, len(idx_post)),
        },
        index=idx_post,
    )
    save_parquet(de_lu, smard_dir / "DE-LU.parquet", downcast=False)

    # Create synthetic DE-AT-LU data (pre-split)
    idx_pre = pd.date_range("2015-01-01", "2018-12-31", freq="h", tz="UTC")
    de_at_lu = pd.DataFrame(
        {
            "marktpreis_deutschland_oesterreich_luxemburg": rng.normal(40, 10, len(idx_pre)),
            "stromerzeugung_wind_onshore": rng.uniform(1000, 20000, len(idx_pre)),
            "stromerzeugung_photovoltaik": rng.uniform(0, 15000, len(idx_pre)),
            "stromverbrauch_gesamt_(netzlast)": rng.uniform(40000, 80000, len(idx_pre)),
        },
        index=idx_pre,
    )
    save_parquet(de_at_lu, smard_dir / "DE-AT-LU.parquet", downcast=False)

    output_path = tmp_path / "merged.parquet"

    from energy_forecasting.data.merge import run_merge_pipeline

    result = run_merge_pipeline(
        smard_dir=smard_dir,
        commodities_dir=tmp_path / "commodities",  # doesn't exist, will skip
        ec_dir=tmp_path / "ec",  # doesn't exist, will skip
        output_path=output_path,
        tso_output_dir=tmp_path / "tso_out",
    )

    # Basic checks
    assert output_path.exists()
    assert "target_price" in result.columns
    assert "regime_de_at_lu" in result.columns
    assert "regime_quarter_hourly" in result.columns
    # Output is tz-naive local time (Europe/Berlin delivery hours)
    assert result.index.tz is None

    # Regime indicators should be correct
    # BIDDING_AREA_SPLIT = 2018-09-30 22:00 UTC = 2018-10-01 00:00 CET
    local_split = BIDDING_AREA_SPLIT.tz_convert("Europe/Berlin").tz_localize(None)
    pre_split = result[result.index < local_split]
    if not pre_split.empty:
        assert (pre_split["regime_de_at_lu"] == 1).all()
