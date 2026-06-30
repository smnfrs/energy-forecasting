"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """A small DataFrame with typical column types for round-trip testing."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "float_col": rng.normal(50.0, 10.0, n),
            "int_col": rng.integers(0, 1000, n),
            "small_float": rng.uniform(0, 1, n),
            "all_nan": np.full(n, np.nan),
        }
    )


@pytest.fixture
def hourly_index():
    """DatetimeIndex covering 1 year at hourly frequency."""
    return pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
