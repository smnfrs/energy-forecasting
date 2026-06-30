"""Pydantic response models for the forecast API.

Defines the output shape that inference produces and the API serves.
Ported from EP's src/api/schemas.py, extended for gen/load targets
and prediction intervals.
"""

from datetime import datetime

from pydantic import BaseModel

# ── Forecast schemas ─────────────────────────────────────────────────


class HourlyForecast(BaseModel):
    """Single hourly forecast point with optional prediction interval."""

    timestamp: datetime
    forecast: float
    forecast_lower: float | None = None
    forecast_upper: float | None = None


class ForecastResponse(BaseModel):
    """Complete forecast for a single target and region."""

    target: str  # "price", "wind_onshore", "solar", "load"
    region: str  # "DE_LU", "DE_50HZ", "national", etc.
    issued_at: datetime
    horizon_hours: int  # 24 for price, 168 for gen/load
    unit: str  # "EUR/MWh", "MW"
    forecasts: list[HourlyForecast]
    ensemble_method: str | None = None  # "blend" or "stacking"
    model_count: int | None = None


class ForecastHistoryResponse(BaseModel):
    """Historical forecasts for review/backtesting."""

    target: str
    forecasts: list[ForecastResponse]
    count: int


# ── Model/performance schemas ────────────────────────────────────────


class ModelInfo(BaseModel):
    """Individual model in an ensemble."""

    name: str
    category: str  # "lgbm", "xgboost", "catboost", "linear"
    weight: float


class ModelsResponse(BaseModel):
    """Ensemble composition and metrics."""

    target: str
    ensemble_method: str
    models: list[ModelInfo]
    holdout_mae: float
    holdout_rmse: float
    pi_coverage: float | None = None
    last_retrain: datetime


class DailyError(BaseModel):
    """One day's forecast error."""

    date: datetime
    mae: float
    rmse: float


class PerformanceResponse(BaseModel):
    """Forecast accuracy over time."""

    target: str
    blend_errors: list[DailyError]
    pi_coverage_30d: float | None = None


# ── Health ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """API health check."""

    status: str  # "healthy", "degraded", "unhealthy"
    models_loaded: int
    data_available: bool
    last_data_update: datetime | None = None
