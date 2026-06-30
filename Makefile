.PHONY: help env install lint format test mlflow serve clean data update data-smard data-weather data-commodities process features-slim features-full features-max features-validate train-gen-load train-gen-load-quick train-gen-load-target

CONDA_ENV ?= energy-forecasting

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

env:  ## Create conda env '$(CONDA_ENV)' (Python 3.13) and install deps from pyproject.toml
	conda create -n $(CONDA_ENV) python=3.13 -y
	conda run -n $(CONDA_ENV) pip install -e ".[dev]"

install:  ## Install package in editable mode with dev deps (into active env)
	pip install -e ".[dev]"

lint:  ## Check formatting and lint
	ruff check energy_forecasting/ tests/
	ruff format --check energy_forecasting/ tests/

format:  ## Auto-format and fix
	ruff check --fix energy_forecasting/ tests/
	ruff format energy_forecasting/ tests/

test:  ## Run tests
	pytest tests/ -v

mlflow:  ## Start MLflow UI
	mlflow ui --backend-store-uri sqlite:///models/mlflow.db --port 5000

serve:  ## Start FastAPI dev server
	uvicorn energy_forecasting.api.app:app --reload --port 8000

clean:  ## Remove compiled files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# ── Data targets ────────────────────────────────────────────────────

data:  ## Download all data from scratch
	energy-forecasting download all

update:  ## Incremental update all sources
	energy-forecasting update all

data-smard:  ## Download SMARD only (national + per-TSO)
	energy-forecasting download smard --region DE-LU
	energy-forecasting download smard --region DE-AT-LU
	energy-forecasting download smard-tso --tso 50Hertz
	energy-forecasting download smard-tso --tso Amprion
	energy-forecasting download smard-tso --tso TenneT
	energy-forecasting download smard-tso --tso TransnetBW
	energy-forecasting download smard-tso --tso Creos

data-weather:  ## Download weather only (all types x TSOs)
	energy-forecasting download weather --all

data-commodities:  ## Download commodities only
	energy-forecasting download commodities

# ── Stage 3 targets ──────────────────────────────────────────────────

process:  ## Clean and merge -> processed/merged.parquet + processed/tso/
	energy-forecasting process

# ── Stage 4 targets ──────────────────────────────────────────────────

features-slim:  ## Compute slim price feature matrix
	energy-forecasting features --feature-list slim

features-full:  ## Compute full price feature matrix
	energy-forecasting features --feature-list full

features-max:  ## Compute max price feature matrix (stage 5c — feature-selection input)
	energy-forecasting features --feature-list max

features-validate:  ## Validate feature lists (no computation)
	energy-forecasting features --feature-list slim --validate-only
	energy-forecasting features --feature-list full --validate-only
	energy-forecasting features --feature-list max --validate-only
	energy-forecasting features --feature-list gen_load --validate-only

# ── Stage 5 targets ─────────────────────────────────────────────────

train-gen-load:  ## Train all gen/load models (70 trials, all targets/regions)
	energy-forecasting train gen-load

train-gen-load-quick:  ## Quick gen/load training (10 trials, for validation)
	energy-forecasting train gen-load --trials 10

train-gen-load-target:  ## Train one target (usage: make train-gen-load-target TARGET=wind_onshore)
	energy-forecasting train gen-load --target $(TARGET) --trials 70

# ── Stage 6+ targets (stubs, implemented in later stages) ───────────
# forecast:   ## Run daily inference
# retrain:    ## Full retrain pipeline
# sync:       ## Pull latest data from GitHub Release
