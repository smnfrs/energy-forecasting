"""FastAPI endpoint handlers.

All endpoints serve pre-computed JSON produced by the daily inference pipeline.
No model inference happens at request time.

Endpoints:
    GET /health                         API health
    GET /forecast/price                 24h D+1 price forecast
    GET /forecast/generation/{type}     168h national generation forecast
    GET /forecast/load                  168h national load forecast
    GET /forecast/history               Rolling 30-day price history
    GET /models                         Ensemble composition
    GET /models/performance             Daily forecast errors
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from energy_forecasting.api import dependencies as deps
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

router = APIRouter()


def _parse_forecast_response(raw: dict) -> ForecastResponse:
    """Convert a raw JSON dict to a ForecastResponse."""
    forecasts = [
        HourlyForecast(
            timestamp=datetime.fromisoformat(f["timestamp"]),
            forecast=f["forecast"],
            forecast_lower=f.get("forecast_lower"),
            forecast_upper=f.get("forecast_upper"),
        )
        for f in raw.get("forecasts", [])
    ]
    return ForecastResponse(
        target=raw["target"],
        region=raw["region"],
        issued_at=datetime.fromisoformat(raw["issued_at"].replace("Z", "+00:00")),
        horizon_hours=raw.get("horizon_hours", len(forecasts)),
        unit=raw.get("unit", ""),
        forecasts=forecasts,
        ensemble_method=raw.get("ensemble_method"),
        model_count=raw.get("model_count"),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models_loaded = deps.count_model_files()
    data_available = deps.is_data_available()

    last_data_update = None
    if data_available:
        try:
            raw = deps.get_price_forecast()
            last_data_update = datetime.fromisoformat(raw["issued_at"].replace("Z", "+00:00"))
        except Exception:
            pass

    if models_loaded > 0 and data_available:
        status = "healthy"
    elif data_available:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        data_available=data_available,
        last_data_update=last_data_update,
    )


@router.get("/forecast/price", response_model=ForecastResponse)
def price_forecast() -> ForecastResponse:
    try:
        return _parse_forecast_response(deps.get_price_forecast())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Price forecast not yet available")


@router.get("/forecast/generation/{gen_type}", response_model=ForecastResponse)
def generation_forecast(gen_type: str) -> ForecastResponse:
    valid = {"wind_onshore", "wind_offshore", "solar"}
    if gen_type not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid generation type '{gen_type}'. Choose from: {sorted(valid)}",
        )
    try:
        return _parse_forecast_response(deps.get_gen_load_forecast(gen_type))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"{gen_type} forecast not yet available")


@router.get("/forecast/load", response_model=ForecastResponse)
def load_forecast() -> ForecastResponse:
    try:
        return _parse_forecast_response(deps.get_gen_load_forecast("load"))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Load forecast not yet available")


@router.get("/forecast/history", response_model=ForecastHistoryResponse)
def forecast_history() -> ForecastHistoryResponse:
    try:
        raw = deps.get_forecast_history()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Forecast history not yet available")

    forecasts = [_parse_forecast_response(f) for f in raw.get("forecasts", [])]
    return ForecastHistoryResponse(
        target="price",
        forecasts=forecasts,
        count=len(forecasts),
    )


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    try:
        meta = deps.get_model_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model metadata not yet available")

    model_list = [
        ModelInfo(name=m["name"], category=m["category"], weight=m["weight"])
        for m in meta.get("models", [])
    ]
    return ModelsResponse(
        target=meta.get("target", "price"),
        ensemble_method=meta.get("ensemble_method", "slsqp_optimized"),
        models=model_list,
        holdout_mae=meta.get("holdout_mae", 0.0),
        holdout_rmse=meta.get("holdout_rmse", 0.0),
        pi_coverage=meta.get("pi_coverage"),
        last_retrain=datetime.fromisoformat(meta["last_retrain"].replace("Z", "+00:00")),
    )


@router.get("/models/performance", response_model=PerformanceResponse)
def performance(days: int | None = None) -> PerformanceResponse:
    errors = deps.get_daily_errors(days=days)
    blend_errors = [
        DailyError(
            date=datetime.fromisoformat(e["date"]),
            mae=e["mae"],
            rmse=e["rmse"],
        )
        for e in errors
    ]
    return PerformanceResponse(
        target="price",
        blend_errors=blend_errors,
    )
