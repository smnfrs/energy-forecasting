"""Integration tests for data coverage — run after full download.

These tests require real data on disk. Skip if data directory doesn't exist.
Run with: pytest tests/test_data_coverage.py -v
"""

import pandas as pd
import pytest
from energy_forecasting.config import COMMODITIES_DIR, SMARD_DIR, WEATHER_DIR
from energy_forecasting.config.smard import KNOWN_MISSING, TSO_FILTER_KEYS, TSO_REGIONS
from energy_forecasting.data.io import load_parquet
from energy_forecasting.data.weather import PHYSICAL_LIMITS

# Skip all tests if raw data doesn't exist
pytestmark = pytest.mark.skipif(
    not SMARD_DIR.exists(),
    reason="Raw data not downloaded — run 'make data' first",
)


class TestSmardCoverage:
    def test_national_de_lu_exists(self):
        path = SMARD_DIR / "DE-LU.parquet"
        if not path.exists():
            pytest.skip("DE-LU not downloaded")
        df = load_parquet(path)
        assert len(df.columns) > 30, f"Expected 30+ columns, got {len(df.columns)}"
        assert df.index.tz is not None, "Index should be UTC-aware"

    def test_tso_files_have_correct_columns(self):
        for tso_name in TSO_REGIONS:
            path = SMARD_DIR / "tso" / f"{tso_name}.parquet"
            if not path.exists():
                continue
            df = load_parquet(path)
            n_missing = sum(1 for fid, r in KNOWN_MISSING if r == tso_name)
            expected_cols = len(TSO_FILTER_KEYS) - n_missing
            assert len(df.columns) == expected_cols, (
                f"{tso_name}: expected {expected_cols} columns, got {len(df.columns)}"
            )

    def test_all_parquets_utc_aware(self):
        for path in SMARD_DIR.rglob("*.parquet"):
            df = load_parquet(path)
            assert df.index.tz is not None, f"{path.name} index not UTC-aware"


class TestCommodityCoverage:
    @pytest.mark.parametrize(
        "name,expected_start_year",
        [
            ("icap.parquet", 2014),
            ("ttf.parquet", 2017),
            ("brent.parquet", 2015),
        ],
    )
    def test_commodity_start_date(self, name, expected_start_year):
        path = COMMODITIES_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not downloaded")
        df = load_parquet(path)
        assert df.index.min().year <= expected_start_year

    def test_no_large_gaps(self):
        """No commodity should have an unexpected source-specific gap."""
        max_gap_by_source = {
            "fred_eu_gas.parquet": pd.Timedelta(days=35),  # monthly FRED cadence
        }
        default_max_gap = pd.Timedelta(days=30)
        for path in COMMODITIES_DIR.glob("*.parquet"):
            df = load_parquet(path)
            if len(df) < 2:
                continue
            gaps = df.index.to_series().diff()
            max_gap = gaps.max()
            allowed = max_gap_by_source.get(path.name, default_max_gap)
            assert max_gap <= allowed, f"{path.name} has a {max_gap.days}-day gap"


class TestWeatherCoverage:
    def test_weather_within_physical_limits(self):
        """Sample check: weather values should be within physical bounds."""
        for path in WEATHER_DIR.rglob("history.parquet"):
            df = load_parquet(path)
            for var, (lo, hi) in PHYSICAL_LIMITS.items():
                matching = [c for c in df.columns if c.startswith(var)]
                for col in matching:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    assert col_min >= lo, f"{col} min {col_min} < {lo}"
                    assert col_max <= hi, f"{col} max {col_max} > {hi}"
