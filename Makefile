.PHONY: help env install lint format test mlflow serve clean data update data-smard data-weather data-commodities process features-slim features-full features-max features-validate train-gen-load train-gen-load-quick train-gen-load-target forecast forecast-skip-update export-models gen-load-config retrain retrain-gen-load sync serve-dashboard open-dashboard test-dashboard

CONDA_ENV ?= energy-forecasting
PYTHON    := conda run -n $(CONDA_ENV) python
EF        := conda run -n $(CONDA_ENV) energy-forecasting

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

env:  ## Create conda env '$(CONDA_ENV)' (Python 3.13) and install deps from pyproject.toml
	conda create -n $(CONDA_ENV) python=3.13 -y
	conda run -n $(CONDA_ENV) pip install -e ".[dev]"

install:  ## Install package in editable mode with dev deps (into active env)
	pip install -e ".[dev]"

lint:  ## Check formatting and lint
	conda run -n $(CONDA_ENV) ruff check energy_forecasting/ tests/
	conda run -n $(CONDA_ENV) ruff format --check energy_forecasting/ tests/

format:  ## Auto-format and fix
	conda run -n $(CONDA_ENV) ruff check --fix energy_forecasting/ tests/
	conda run -n $(CONDA_ENV) ruff format energy_forecasting/ tests/

test:  ## Run tests
	conda run -n $(CONDA_ENV) pytest tests/ -v

mlflow:  ## Start MLflow UI
	conda run -n $(CONDA_ENV) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:  ## Start FastAPI dev server (Stage 6/7 API/dashboard surface)
	conda run -n $(CONDA_ENV) uvicorn energy_forecasting.api.app:app --reload --port 8000

clean:  ## Remove compiled files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# ── Data targets ────────────────────────────────────────────────────────────

data:  ## Download all data from scratch
	$(EF) download all

update:  ## Incremental update all sources
	$(EF) update all

data-smard:  ## Download SMARD only (national + per-TSO)
	$(EF) download smard --region DE-LU
	$(EF) download smard --region DE-AT-LU
	$(EF) download smard-tso --tso 50Hertz
	$(EF) download smard-tso --tso Amprion
	$(EF) download smard-tso --tso TenneT
	$(EF) download smard-tso --tso TransnetBW
	$(EF) download smard-tso --tso Creos

data-weather:  ## Download weather only (all types x TSOs)
	$(EF) download weather --all

data-commodities:  ## Download commodities only
	$(EF) download commodities

# ── Stage 3 targets ──────────────────────────────────────────────────────────

process:  ## Clean and merge -> processed/merged.parquet + processed/tso/
	$(EF) process

# ── Stage 4 targets ──────────────────────────────────────────────────────────

features-slim:  ## Compute slim price feature matrix
	$(EF) features --feature-list slim

features-full:  ## Compute full price feature matrix
	$(EF) features --feature-list full

features-max:  ## Compute max price feature matrix (stage 5c — feature-selection input)
	$(EF) features --feature-list max

features-validate:  ## Validate feature lists (no computation)
	$(EF) features --feature-list slim --validate-only
	$(EF) features --feature-list full --validate-only
	$(EF) features --feature-list max --validate-only
	$(EF) features --feature-list gen_load --validate-only

# ── Stage 5 targets ──────────────────────────────────────────────────────────

train-gen-load:  ## Train all gen/load models (70 trials, all targets/regions)
	$(EF) train gen-load

train-gen-load-quick:  ## Quick gen/load training (10 trials, for validation)
	$(EF) train gen-load --trials 10

train-gen-load-target:  ## Train one target (usage: make train-gen-load-target TARGET=wind_onshore)
	$(EF) train gen-load --target $(TARGET) --trials 70

# ── Stage 6 targets ──────────────────────────────────────────────────────────

forecast:  ## Run full daily inference pipeline (data update → gen/load → price → output)
	$(EF) deploy forecast

forecast-skip-update:  ## Run inference only (skip data update, use existing data)
	$(EF) deploy forecast --skip-update

export-models:  ## Export production models from MLflow to disk (models/gen_load/, models/price/)
	$(EF) deploy export-models

gen-load-config:  ## Scan MLflow and write models/gen_load_config.json from best runs
	$(EF) deploy gen-load-config

retrain:  ## Retrain price ensemble from stored hyperparams (price only — CI-safe ~30-90 min)
	$(EF) deploy retrain --price-only

retrain-gen-load:  ## Full gen/load retrain detached (8-12 hours — runs on the tower)
	setsid nohup bash -c '$(EF) deploy retrain' </dev/null > logs/retrain_gen_load.log 2>&1 & \
	disown && echo "Gen/load retrain PID=$$! — tail logs/retrain_gen_load.log"

sync:  ## Pull latest models from GitHub Release
	gh release download latest --dir models/ --clobber

# ── Stage 7 targets ──────────────────────────────────────────────────────────

serve-dashboard:  ## Serve the static dashboard locally on port 8080
	cd deploy && $(PYTHON) -m http.server 8080

open-dashboard:  ## Open the dashboard in the default browser
	xdg-open http://localhost:8080

test-dashboard:  ## Run Playwright smoke tests (requires serve-dashboard running)
	npx playwright test tests/test_dashboard.js --headed
