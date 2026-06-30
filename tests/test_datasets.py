"""Tests for modeling/datasets.py."""

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.datasets import (
    TARGET_COL_SUFFIX,
    find_dataset,
    load_dataset,
)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal Parquet dataset file."""
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "feature_a": np.random.default_rng(42).normal(size=100),
            "feature_b": np.random.default_rng(43).normal(size=100),
            f"price{TARGET_COL_SUFFIX}": np.random.default_rng(44).normal(50, 10, 100),
        },
        index=idx,
    )
    path = tmp_path / "test_dataset.parquet"
    df.to_parquet(path)
    return path


class TestLoadDataset:
    def test_splits_x_and_y(self, sample_dataset):
        X, y = load_dataset(sample_dataset)
        assert "feature_a" in X.columns
        assert "feature_b" in X.columns
        # Target column should not be in X
        assert not any(c.endswith(TARGET_COL_SUFFIX) for c in X.columns)
        assert y.name == "price"
        assert len(y) == 100

    def test_no_target_column_raises(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "no_target.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="target column"):
            load_dataset(path)


class TestFindDataset:
    def test_missing_returns_none(self):
        result = find_dataset("nonexistent_dataset_xyz_999")
        assert result is None
