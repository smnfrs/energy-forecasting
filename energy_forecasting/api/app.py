"""FastAPI application for the Energy Forecasting API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from energy_forecasting.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from energy_forecasting.api import dependencies as deps

    app.state.models_loaded = deps.count_model_files()
    app.state.data_available = deps.is_data_available()
    yield


app = FastAPI(
    title="Energy Forecasting API",
    description=(
        "Day-ahead electricity price and generation/load forecasts "
        "for the German energy market (DE/LU)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)
