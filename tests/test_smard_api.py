"""Tests for data/smard.py — SMARD API client with mocked HTTP."""

import pandas as pd
import pytest
import responses
from energy_forecasting.config.smard import SMARD_API_BASE
from energy_forecasting.data.smard import (
    DataNotAvailableError,
    get_all_data,
    get_data,
    get_timestamps,
)


@pytest.fixture
def mock_timestamps():
    """Sample weekly-chunk timestamps (ms)."""
    return [1704067200000, 1704672000000, 1705276800000]  # ~weekly apart


@pytest.fixture
def mock_series_data():
    """Sample series data for one chunk (3 hourly rows)."""
    base = 1704067200000  # 2024-01-01 00:00 UTC
    hour_ms = 3600000
    return [
        [base, 42.5],
        [base + hour_ms, 43.1],
        [base + 2 * hour_ms, None],  # NaN row that should be dropped
    ]


@responses.activate
def test_get_timestamps_success(mock_timestamps):
    url = f"{SMARD_API_BASE}/4169/DE-LU/index_hour.json"
    responses.add(
        responses.GET,
        url,
        json={"timestamps": mock_timestamps},
        status=200,
    )
    result = get_timestamps(4169, "DE-LU", "hour")
    assert result == mock_timestamps


@responses.activate
def test_get_timestamps_404():
    url = f"{SMARD_API_BASE}/9999/DE-LU/index_hour.json"
    responses.add(responses.GET, url, status=404)
    with pytest.raises(DataNotAvailableError):
        get_timestamps(9999, "DE-LU", "hour")


@responses.activate
def test_get_data_parses_response(mock_series_data):
    ts = 1704067200000
    url = f"{SMARD_API_BASE}/4169/DE-LU/4169_DE-LU_hour_{ts}.json"
    responses.add(
        responses.GET,
        url,
        json={"series": mock_series_data},
        status=200,
    )
    df = get_data(4169, "DE-LU", ts, "hour")
    assert "timestamp" in df.columns
    assert "value" in df.columns
    assert "time" in df.columns
    assert len(df) == 3  # Including NaN row (not dropped by get_data)
    assert df["time"].dt.tz is not None  # UTC-aware


@responses.activate
def test_get_all_data_concatenates_and_deduplicates(mock_timestamps, mock_series_data):
    # Register timestamp index
    idx_url = f"{SMARD_API_BASE}/4169/DE-LU/index_hour.json"
    responses.add(
        responses.GET,
        idx_url,
        json={"timestamps": mock_timestamps},
        status=200,
    )
    # Register data for each timestamp (reuse same data — the dedup logic matters)
    for ts in mock_timestamps:
        url = f"{SMARD_API_BASE}/4169/DE-LU/4169_DE-LU_hour_{ts}.json"
        responses.add(
            responses.GET,
            url,
            json={"series": mock_series_data},
            status=200,
        )
    df = get_all_data(4169, "DE-LU", "hour")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert "value" in df.columns
    # NaN rows should be dropped
    assert df["value"].notna().all()


@responses.activate
def test_get_all_data_with_explicit_timestamps(mock_series_data):
    ts = 1704067200000
    url = f"{SMARD_API_BASE}/4169/DE-LU/4169_DE-LU_hour_{ts}.json"
    responses.add(
        responses.GET,
        url,
        json={"series": mock_series_data},
        status=200,
    )
    df = get_all_data(4169, "DE-LU", "hour", timestamp_list=[ts])
    assert not df.empty
    assert df["value"].notna().all()


def test_get_all_data_empty_timestamp_list():
    df = get_all_data(4169, "DE-LU", "hour", timestamp_list=[])
    assert df.empty
