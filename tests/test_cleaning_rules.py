"""Tests for config/cleaning.py and data/processing.py helper functions."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.data.processing import (
    clip_bounds,
    drop_columns,
    fill_from_column,
    fill_from_difference,
    fill_zero_after,
    fill_zero_before,
    fill_zero_before_first_valid,
    interpolate_gaps,
)


@pytest.fixture
def ts_df():
    """DataFrame with hourly DatetimeIndex and some test columns."""
    idx = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "temperature_2m_berlin": rng.normal(10, 5, 100),
            "wind_speed_north": rng.uniform(0, 30, 100),
            "target_price": rng.normal(50, 20, 100),
            "stromverbrauch_gesamt": rng.uniform(40000, 80000, 100),
            "col_a": rng.normal(0, 1, 100),
            "col_b": rng.normal(0, 1, 100),
        },
        index=idx,
    )


def test_drop_columns_existing(ts_df):
    result = drop_columns(ts_df, ["col_a", "col_b"])
    assert "col_a" not in result.columns
    assert "col_b" not in result.columns
    assert "target_price" in result.columns


def test_drop_columns_missing_ignored(ts_df):
    result = drop_columns(ts_df, ["nonexistent", "col_a"])
    assert "col_a" not in result.columns
    assert len(result.columns) == len(ts_df.columns) - 1


def test_clip_bounds_clips_values(ts_df):
    result = clip_bounds(ts_df, "temperature_2m_*", min_val=-10, max_val=20)
    assert result["temperature_2m_berlin"].min() >= -10
    assert result["temperature_2m_berlin"].max() <= 20


def test_clip_bounds_nan_action(ts_df):
    ts_df.loc[ts_df.index[0], "target_price"] = 2000  # out of bounds
    result = clip_bounds(ts_df, "target_price", min_val=-500, max_val=1000, action="nan")
    assert pd.isna(result.loc[result.index[0], "target_price"])


def test_clip_bounds_wildcard_no_match(ts_df):
    # Should not error on no matches
    result = clip_bounds(ts_df, "nonexistent_*", min_val=0, max_val=100)
    pd.testing.assert_frame_equal(result, ts_df)


def test_fill_zero_after_cutoff():
    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"x": [1, 2, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan, np.nan]}, index=idx
    )
    cutoff = str(idx[4])
    result = fill_zero_after(df, "x", after=cutoff)
    # NaN before cutoff should remain
    assert pd.isna(result.loc[idx[2], "x"])
    # NaN after cutoff should be 0
    assert result.loc[idx[6], "x"] == 0
    assert result.loc[idx[9], "x"] == 0


def test_fill_zero_after_last_valid():
    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"x": [1, 2, 3, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan]}, index=idx
    )
    result = fill_zero_after(df, "x", after="last_valid")
    # Last valid is idx[5] (value=5)
    assert result.loc[idx[6], "x"] == 0
    assert result.loc[idx[9], "x"] == 0
    # NaN before last valid should remain
    assert pd.isna(result.loc[idx[3], "x"])


def test_fill_zero_before():
    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, np.nan, 9]}, index=idx)
    cutoff = str(idx[3])
    result = fill_zero_before(df, "x", before=cutoff)
    assert result.loc[idx[0], "x"] == 0
    assert result.loc[idx[2], "x"] == 0
    # After cutoff: NaN stays NaN
    assert pd.isna(result.loc[idx[8], "x"])


def test_fill_zero_before_first_valid():
    idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, 8, 9]}, index=idx)
    result = fill_zero_before_first_valid(df, "x")
    assert result.loc[idx[0], "x"] == 0
    assert result.loc[idx[2], "x"] == 0
    assert result.loc[idx[3], "x"] == 3  # first valid preserved


def test_fill_from_difference():
    df = pd.DataFrame(
        {
            "target": [np.nan, 10, np.nan],
            "total": [100, 100, 100],
            "subtract": [30, 30, 30],
        }
    )
    result = fill_from_difference(df, "target", total="total", subtract="subtract")
    assert result.loc[0, "target"] == 70
    assert result.loc[1, "target"] == 10  # already had value, but was overwritten by NaN fill
    assert result.loc[2, "target"] == 70


def test_fill_from_column():
    df = pd.DataFrame(
        {
            "target": [np.nan, 2, np.nan],
            "source": [10, 20, 30],
        }
    )
    result = fill_from_column(df, "target", source="source")
    assert result.loc[0, "target"] == 10
    assert result.loc[1, "target"] == 2  # preserved
    assert result.loc[2, "target"] == 30


def test_interpolate_gaps_respects_max_gap():
    idx = pd.date_range("2023-01-01", periods=20, freq="h")
    values = list(range(20))
    # Create a gap of 3 (should be filled with max_gap=5)
    values[5] = np.nan
    values[6] = np.nan
    values[7] = np.nan
    # Create a gap of 7 (should NOT be filled with max_gap=5)
    values[12] = np.nan
    values[13] = np.nan
    values[14] = np.nan
    values[15] = np.nan
    values[16] = np.nan
    values[17] = np.nan
    values[18] = np.nan

    df = pd.DataFrame({"x": values}, index=idx, dtype=float)
    result = interpolate_gaps(df, method="linear", max_gap=5)

    # Small gap should be filled
    assert not pd.isna(result.loc[idx[5], "x"])
    assert not pd.isna(result.loc[idx[7], "x"])

    # Large gap should remain NaN
    assert pd.isna(result.loc[idx[14], "x"])


def test_interpolate_gaps_excludes_columns():
    idx = pd.date_range("2023-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {
            "x": [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10],
            "regime": [0, np.nan, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        index=idx,
        dtype=float,
    )
    result = interpolate_gaps(df, method="linear", max_gap=5, exclude=["regime"])
    assert not pd.isna(result.loc[idx[1], "x"])
    assert pd.isna(result.loc[idx[1], "regime"])


def test_clean_column_name_transliterates_umlauts():
    from energy_forecasting.config.columns import clean_column_name

    assert clean_column_name("Marktpreis: Österreich") == "marktpreis_oesterreich"
    assert clean_column_name("Marktpreis: Deutschland/Österreich/Luxemburg") == (
        "marktpreis_deutschland_oesterreich_luxemburg"
    )
