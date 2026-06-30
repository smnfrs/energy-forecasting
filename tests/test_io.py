"""Tests for data/io.py — save_parquet + load_parquet round-trip."""

import numpy as np
import pandas as pd
from energy_forecasting.data.io import load_parquet, reduce_dtypes, save_parquet


def test_reduce_dtypes_downcasts_float64(sample_df):
    result = reduce_dtypes(sample_df)
    assert result["float_col"].dtype == np.float32
    assert result["small_float"].dtype == np.float32


def test_reduce_dtypes_downcasts_int64(sample_df):
    result = reduce_dtypes(sample_df)
    assert result["int_col"].dtype == np.int32


def test_reduce_dtypes_skips_all_nan(sample_df):
    result = reduce_dtypes(sample_df)
    # all-NaN columns are skipped (stay float64)
    assert result["all_nan"].dtype == np.float64


def test_reduce_dtypes_does_not_mutate(sample_df):
    original_dtype = sample_df["float_col"].dtype
    reduce_dtypes(sample_df)
    assert sample_df["float_col"].dtype == original_dtype


def test_round_trip_with_downcasting(sample_df, tmp_path):
    path = tmp_path / "test.parquet"
    save_parquet(sample_df, path, compress=True, downcast=True)
    loaded = load_parquet(path)

    assert len(loaded) == len(sample_df)
    assert set(loaded.columns) == set(sample_df.columns)
    # Values should match within float32 tolerance
    np.testing.assert_allclose(
        loaded["float_col"].values,
        sample_df["float_col"].values,
        rtol=1e-6,
    )


def test_round_trip_without_downcasting(sample_df, tmp_path):
    path = tmp_path / "test.parquet"
    save_parquet(sample_df, path, compress=False, downcast=False)
    loaded = load_parquet(path)

    assert loaded["float_col"].dtype == np.float64
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_zstd_compression_smaller(sample_df, tmp_path):
    compressed_path = tmp_path / "compressed.parquet"
    uncompressed_path = tmp_path / "uncompressed.parquet"

    save_parquet(sample_df, compressed_path, compress=True, downcast=False)
    save_parquet(sample_df, uncompressed_path, compress=False, downcast=False)

    assert compressed_path.stat().st_size < uncompressed_path.stat().st_size


def test_save_creates_parent_dirs(sample_df, tmp_path):
    path = tmp_path / "sub" / "dir" / "test.parquet"
    save_parquet(sample_df, path)
    assert path.exists()
