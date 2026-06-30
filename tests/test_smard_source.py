"""Tests for SmardSource in data/sources.py — mocked API."""

import pandas as pd
import pytest
import responses
from energy_forecasting.config.smard import KNOWN_MISSING, SMARD_API_BASE, TSO_FILTER_KEYS
from energy_forecasting.data.sources import SmardSource, _merge_column

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_chunk():
    """A minimal valid SMARD series response (2 non-null rows)."""
    base = 1704067200000
    hour_ms = 3600000
    return {
        "series": [
            [base, 100.0],
            [base + hour_ms, 200.0],
        ]
    }


def _register_filter(filter_id, region, mock_chunk, timestamps=None):
    """Register mocked responses for one filter/region combo."""
    if timestamps is None:
        timestamps = [1704067200000]
    idx_url = f"{SMARD_API_BASE}/{filter_id}/{region}/index_hour.json"
    responses.add(
        responses.GET,
        idx_url,
        json={"timestamps": timestamps},
        status=200,
    )
    for ts in timestamps:
        data_url = f"{SMARD_API_BASE}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json"
        responses.add(responses.GET, data_url, json=mock_chunk, status=200)


# ── Tests ───────────────────────────────────────────────────────────


def test_smard_source_national_properties():
    src = SmardSource("DE-LU")
    assert not src._is_tso
    assert "DE_LU" in str(src.output_path) or "DE-LU" in str(src.output_path)
    # Should have many filter keys (national = SMARD + cross-border)
    assert len(src.filter_keys) > 30


def test_smard_source_tso_properties():
    src = SmardSource("50Hertz")
    assert src._is_tso
    assert "tso" in str(src.output_path)
    # TSO filter keys minus known missing
    n_missing = sum(1 for fid, r in KNOWN_MISSING if r == "50Hertz")
    assert len(src.filter_keys) == len(TSO_FILTER_KEYS) - n_missing


def test_smard_source_tso_column_names():
    src = SmardSource("Amprion")
    for fid, name in src.filter_keys.items():
        col = src._column_name(fid, name)
        assert col.endswith("_ampr"), f"Expected _ampr suffix, got {col}"


def test_known_missing_excluded_from_filter_keys():
    """KNOWN_MISSING combos should not appear in filter_keys."""
    for fid, region in KNOWN_MISSING:
        src = SmardSource(region)
        assert fid not in src.filter_keys


@responses.activate
def test_smard_source_download_creates_file(tmp_path, mock_chunk, monkeypatch):
    """SmardSource.download() creates a Parquet with expected columns."""
    # Use a TSO source with fewer keys for faster test
    monkeypatch.setattr(
        "energy_forecasting.config.smard.TSO_FILTER_KEYS",
        {4068: "solar", 410: "load"},
    )
    monkeypatch.setattr(
        "energy_forecasting.config.smard.KNOWN_MISSING",
        set(),
    )
    monkeypatch.setattr(
        "energy_forecasting.data.sources.SmardSource.output_path",
        property(lambda self: tmp_path / "test.parquet"),
    )

    _register_filter(4068, "Amprion", mock_chunk)
    _register_filter(410, "Amprion", mock_chunk)

    src = SmardSource("Amprion")
    src.download()

    assert (tmp_path / "test.parquet").exists()
    df = pd.read_parquet(tmp_path / "test.parquet")
    assert "solar_ampr" in df.columns
    assert "load_ampr" in df.columns
    assert len(df) == 2


# ── _merge_column tests ────────────────────────────────────────────


def test_merge_column_new_column():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
    series = pd.Series([10, 20, 30], index=idx, name="b")
    result = _merge_column(df, "b", series)
    assert "b" in result.columns
    assert len(result) == 3


def test_merge_column_extends_index():
    idx1 = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    idx2 = pd.date_range("2024-01-01 02:00", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"a": [1, 2]}, index=idx1)
    series = pd.Series([30, 40], index=idx2, name="b")
    result = _merge_column(df, "b", series)
    assert len(result) == 4  # Union of both index ranges


def test_merge_column_overwrites_existing():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
    series = pd.Series([10, 20, 30], index=idx, name="a")
    result = _merge_column(df, "a", series)
    assert list(result["a"]) == [10, 20, 30]
