"""Tests for api/schemas.py — Pydantic models instantiate and serialize."""

from datetime import datetime, timezone

import pytest
from energy_forecasting.api.schemas import (
    DailyError,
    ForecastHistoryResponse,
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    ModelInfo,
    ModelsResponse,
    PerformanceResponse,
)


@pytest.fixture
def sample_hourly():
    return HourlyForecast(
        timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        forecast=50.0,
    )


@pytest.fixture
def sample_hourly_with_pi():
    return HourlyForecast(
        timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        forecast=50.0,
        forecast_lower=30.0,
        forecast_upper=70.0,
    )


def test_hourly_forecast_none_intervals(sample_hourly):
    assert sample_hourly.forecast_lower is None
    assert sample_hourly.forecast_upper is None


def test_hourly_forecast_with_intervals(sample_hourly_with_pi):
    assert sample_hourly_with_pi.forecast_lower == 30.0
    assert sample_hourly_with_pi.forecast_upper == 70.0


def test_forecast_response_serializes(sample_hourly):
    resp = ForecastResponse(
        target="price",
        region="DE_LU",
        issued_at=datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc),
        horizon_hours=24,
        unit="EUR/MWh",
        forecasts=[sample_hourly],
    )
    data = resp.model_dump()
    assert data["target"] == "price"
    assert data["region"] == "DE_LU"
    assert data["ensemble_method"] is None
    assert len(data["forecasts"]) == 1


def test_forecast_response_json_roundtrip(sample_hourly):
    resp = ForecastResponse(
        target="price",
        region="DE_LU",
        issued_at=datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc),
        horizon_hours=24,
        unit="EUR/MWh",
        forecasts=[sample_hourly],
        ensemble_method="blend",
        model_count=5,
    )
    json_str = resp.model_dump_json()
    roundtripped = ForecastResponse.model_validate_json(json_str)
    assert roundtripped.target == "price"
    assert roundtripped.model_count == 5


def test_forecast_history_response():
    resp = ForecastHistoryResponse(target="price", forecasts=[], count=0)
    assert resp.count == 0


def test_models_response():
    resp = ModelsResponse(
        target="price",
        ensemble_method="blend",
        models=[ModelInfo(name="lgbm_v1", category="lgbm", weight=0.6)],
        holdout_mae=5.2,
        holdout_rmse=7.1,
        last_retrain=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert resp.models[0].weight == 0.6
    assert resp.pi_coverage is None


def test_performance_response():
    resp = PerformanceResponse(
        target="price",
        blend_errors=[
            DailyError(date=datetime(2024, 1, 1), mae=4.5, rmse=6.0),
        ],
    )
    assert len(resp.blend_errors) == 1
    assert resp.pi_coverage_30d is None


def test_health_response():
    resp = HealthResponse(
        status="healthy",
        models_loaded=5,
        data_available=True,
        last_data_update=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert resp.status == "healthy"


def test_health_response_no_update():
    resp = HealthResponse(
        status="unhealthy",
        models_loaded=0,
        data_available=False,
    )
    assert resp.last_data_update is None
