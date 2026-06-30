# Stage 5: Model Training & Ensembling

**Goal:** Models train, evaluate, and ensemble correctly for all targets. Price models produce day-ahead hourly forecasts with prediction intervals. Gen/load models produce 7-day forecasts per TSO with prediction intervals.

**Split into four sub-stages:**
```
5a: Training Infrastructure  →  5b: Gen/Load Models  →  5c: Price Models  →  5d: Analysis Notebooks
    (config, MLflow, metrics,       (weather FE,              (feature selection,      (diagnostics,
     CV, training loop,              Optuna, per-TSO           joint tuning,            SHAP viz,
     intervals, baselines)           training & ensemble)      ensemble)                residual analysis)
```
5a is a prerequisite for both 5b and 5c. 5b must precede 5c because gen/load Optuna selects weather FE computation choices that feed into price features (load→price FE transfer, master plan §5.4). 5d is deferred until after 5a-5c are complete (notebook generation is token-intensive). Training code in 5a-5c logs the artifacts that 5d notebooks consume.

**Gated execution:** Each sub-stage is a gate. At the end of each sub-stage, stop, present results against the milestones, and wait for user acceptance before starting the next sub-stage. Do not proceed to 5b until 5a is accepted, etc.

---

## Source Material

- EP: `src/modeling/` (training, blend, baselines, metrics, train_final), `src/config/modeling.py`, `notebooks/stacking_blending.ipynb` (9 ensemble methods), `notebooks/feature_selection.ipynb` (SHAP + RFECV), `notebooks/boosted_search.ipynb` (effective hyperparameter ranges), `models/production/blend_hyperparams.json` (production params)
- EMA: `forecasting_modules/` (base_models, tasks, hyperparameters_for_optuna, utils), `data_modules/data_classes.py`
- Already implemented (stages 1-4): `config/modeling.py`, `features/engine.py`, `config/features.py`, weather FE classes, `docs/mlflow_conventions.md`
- Stubs to fill: all files in `modeling/`

---

## Key Design Decisions

### sklearn Pipelines — yes, for the training loop only

EP's mistake was wrapping **feature engineering** in `TransformerMixin` subclasses with versioning. But `Pipeline([StandardScaler(), TransformedTargetRegressor(model)])` for the training loop is standard and useful:
- Composes cleanly with MAPIE `CrossConformalRegressor`
- `clone()` works for CV fold retraining
- MLflow serialization is built-in
- `TransformedTargetRegressor` auto-inverts on `predict()`

### MAPIE on everything

Both price and gen/load models get prediction intervals via MAPIE 1.3 `CrossConformalRegressor`. Ensemble intervals use post-hoc conformal calibration (method-agnostic).

### Direct prediction by default

Follows EMA's approach: `lags_target=None` (direct prediction) is the effective default because honest CV rarely favours lag features. Optuna still searches `{None, 1, 6, 12}` so the comparison is explicit per trial. When a lag is selected, **both CV and holdout use recursive forecasting** (feeding each step's prediction into the lag columns of subsequent rows), mirroring EMA's `forecast_window` and eliminating the train-vs-inference mismatch that would otherwise make `lag_1` a leaky shortcut during CV. Lag-enabled runs bypass MAPIE (PIs reported as NaN) since MAPIE cannot drive the recursive loop. Target-lag columns are named `{target}_h{N}` — same `_h{N}` convention as the TSO-level features (`gen_wind_on_h24`, `load_h24`).

### Datasets as Parquet files, tracked via MLflow

Feature computation (stage 4) is separated from training. `prepare_dataset()` computes features and saves to `data/processed/datasets/{name}.parquet`. Training functions load from Parquet and register datasets with MLflow via `mlflow.log_input()` — datasets appear in each run's Datasets tab, not as separate runs or experiments.

### No magic numbers

All defaults live in `config/modeling.py` with comments explaining where and why they're used.

---

## Pre-requisite Fix: Morning Actuals Cutoff (Stage 4)

Inference runs at **08:00 UTC** (= 09:00 CET winter / 10:00 CEST summer). D-1 is the inference day. Morning actuals from D-1 must be available 1 hour before inference → data available up to **07:00 UTC = 08:00 CET (winter)**.

Current `_eh10` features (hours 0-9) are **not available** in winter. Fix to `_eh7` (hours 0-6):

| Current | Fixed |
|---|---|
| `residual_load_d1_eh10` | `residual_load_d1_eh7` |
| `gen_wind_on_d1_eh10` | `gen_wind_on_d1_eh7` |
| `gen_wind_off_d1_eh10` | `gen_wind_off_d1_eh7` |
| `gen_solar_d1_eh10` | `gen_solar_d1_eh7` |
| `*_ewma_*_d1_h10` | `*_ewma_*_d1_h7` |

EP likely had the same latent bug (also 08:00 UTC inference with `morning_cutoff_cet=10`). Fix in `config/features.py` before stage 5 training begins.

**Scope of fix:** The DSL parser already supports `_eh{N}` generically — no engine changes needed. The fix is:
1. Rename `_eh10` → `_eh7` in `PRICE_FEATURES_SLIM` and `PRICE_FEATURES_FULL` in `config/features.py` (and any EWMA `_h10` → `_h7` variants).
2. Update `SHORT_NAMES` entries if any reference the old suffix.
3. The leakage validator (`features/validation.py`) already checks `end_hour` against availability — update the morning cutoff constant so it catches violations.
4. Add a test that the validation rejects `_eh10` features with the corrected cutoff.
5. No existing datasets need recomputation — this is a config change before any stage 5 datasets are built.

---

# Stage 5a: Training Infrastructure

## 5a.1 Config Extensions

**`energy_forecasting/config/modeling.py`** — extend with all constants, no magic numbers anywhere else:

```python
# ── MAPIE ──────────────────────────────────────────────────────────
# 90% prediction intervals — standard for energy forecasting.
# Used by CrossConformalRegressor in intervals.py.
PI_CONFIDENCE_LEVEL = 0.90

# Internal CV folds for conformal calibration.
# CrossConformalRegressor uses these to compute conformity scores.
PI_CV_FOLDS = 5

# ── Cross-validation ──────────────────────────────────────────────
# CV folds during Optuna search — fewer folds = faster iteration.
# Used by tune_price_model() and tune_gen_load_model() in tuning.py.
SEARCH_CV_FOLDS = 3

# CV folds for final validation of winning models — more folds = better estimate.
# Used by validate_candidates() in ensemble.py and final train_model() calls.
VALIDATION_CV_FOLDS = 5

# ── Holdout ───────────────────────────────────────────────────────
# Days reserved for final evaluation. Carved out BEFORE CV —
# CV never sees holdout data. Used by train_model() in training.py.
HOLDOUT_DAYS = 90

# ── Sample weighting ──────────────────────────────────────────────
# Exponential decay half-life in days. At half_life, weight = 0.5.
# 730 days = 2 years. Used by compute_sample_weights() in training.py.
DEFAULT_WEIGHT_HALF_LIFE = 730.0

# ── Blend candidate selection ─────────────────────────────────────
# Per category (linear, lgbm, xgboost, catboost): 6 candidates each.
# Composition: 2 incumbents (from previous ensemble), 1 best-MAE,
# 1 best-RMSE, 2 random from remaining pool.
# First run (no incumbents): best-MAE, best-RMSE, 4 random.
# Random picks are logged — if they consistently outperform top-metric
# picks in the ensemble, it signals overfitting in the ranked models.
# Candidates are cloned and retrained during validation (VALIDATION_CV_FOLDS each)
# and again for final blend training. 24 candidates × 5 folds = 120 fits
# for validation, plus ~8 final fits — manageable for already-tuned models.
# Used by select_candidates() in ensemble.py.
BLEND_CANDIDATES_PER_CATEGORY = 6
BLEND_CANDIDATES_RANDOM_POOL = 10  # draw 2 random from top-10 after removing incumbents/best

# ── Degradation detection ─────────────────────────────────────────
# If (new_mae - old_mae) / old_mae exceeds this, flag needs_reselection.
# Used by retrain logic in ensemble.py.
BLEND_DEGRADATION_THRESHOLD = 0.20

# ── Ensemble methods to compare at each retrain ───────────────────
# All methods are evaluated on holdout; best is selected.
# See ensemble.py for implementations.
ENSEMBLE_METHODS = [
    "simple_average",
    "inverse_mae",
    "slsqp_optimized",
    "greedy_forward",
    "hill_climbing",
    "simulated_annealing",
    "diversity_regularized",
    "stacking_ridge",
    "stacking_lgbm",
]

# ── Gen/load targets ──────────────────────────────────────────────
GEN_LOAD_TARGETS = {
    "wind_onshore": {"regions": ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNETBW"]},
    "wind_offshore": {"regions": ["DE_50HZ", "DE_TENNET"]},
    "solar": {"regions": ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNETBW"]},
    "load": {"regions": ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNETBW", "DE_CREOS"]},
}
GEN_LOAD_HORIZON_HOURS = 168  # 7 days

# ── Price model categories ────────────────────────────────────────
PEAK_HOURS = list(range(8, 20))
BLEND_CATEGORY_MATCHERS = {
    "linear": ["Ridge", "Lasso", "ElasticNet", "HuberRegressor"],
    "lgbm": ["LGBMRegressor"],
    "xgboost": ["XGBRegressor"],
    "catboost": ["CatBoostRegressor"],
}

# ── MLflow experiments ────────────────────────────────────────────
EXPERIMENTS = {
    "price_feature_selection": "price/feature_selection",
    "price_model_training": "price/model_training",
    "price_production": "price/production",
    "gen_wind_onshore": "generation/wind_onshore",
    "gen_wind_offshore": "generation/wind_offshore",
    "gen_solar": "generation/solar",
    "gen_load": "generation/load",
}

# ── EEG regime dates ──────────────────────────────────────────────
# §51 EEG negative price clawback thresholds.
# Used to create regime indicator features.
EEG_4H_RULE_DATE = "2023-01-01"         # 6h → 4h threshold
EEG_2H_RULE_DATE = "2024-01-01"         # 4h → 2h (interim)
EEG_SOLARSPITZENGESETZ_DATE = "2025-02-25"  # Any negative 15-min block
```

**`energy_forecasting/config/search_spaces.py`** — new file.

Search spaces based on **EP's production parameters** (from `blend_hyperparams.json`). Ranges narrowed around what actually worked — high regularization, moderate learning rates:

```python
def suggest_lgbm(trial) -> dict:
    """LightGBM search space. Based on EP production: lr=0.008-0.02, reg_alpha/lambda=5-7."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 40),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 120),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.75),
        "reg_alpha": trial.suggest_float("reg_alpha", 3.0, 12.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 8.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.05, 0.5),
        "objective": "mae",             # fixed — EP found MAE loss best
        "metric": "mae",                # early stopping monitors MAE
    }

def suggest_xgboost(trial) -> dict:
    """XGBoost. EP production: lr=0.023, reg_alpha=3.8-11.9, gamma=0.4-0.6."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 30, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.75),
        "reg_alpha": trial.suggest_float("reg_alpha", 3.0, 12.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 8.0),
        "gamma": trial.suggest_float("gamma", 0.05, 0.6),
        "objective": "reg:absoluteerror",  # fixed — MAE loss
        "eval_metric": "mae",              # early stopping monitors MAE
    }

def suggest_catboost(trial) -> dict:
    """CatBoost. EP production: lr=0.01-0.02, depth=8-9, l2_leaf_reg=5-7.
    Fixed: verbose=0 (suppress logging), loss_function=MAE.
    """
    return {
        "iterations": trial.suggest_int("iterations", 800, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 8.0),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "rsm": trial.suggest_float("rsm", 0.45, 0.75),  # colsample equivalent
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
        "loss_function": "MAE",  # fixed — EP found MAE loss best
        "eval_metric": "MAE",    # early stopping monitors MAE
        "verbose": 0,            # fixed — suppress CatBoost logging
    }

def suggest_ridge(trial) -> dict:
    """Ridge. EP production: alpha=0.1."""
    return {"alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True)}

def suggest_lasso(trial) -> dict:
    """Lasso. EP production: alpha=0.1."""
    return {"alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True)}

# ── Price model grid points ───────────────────────────────────────
# Two-stage grid for trees:
#   Stage 1: Pin weight_half_life using 1 representative config per model type.
#   Stage 2: Grid over hyperparams with winning weight fixed.
# This avoids weight_half_life multiplying the full hyperparam grid.

# Stage 1: weight selection — 1 full config per type × 4 weights = 12 trials total
# These are EP-production-quality configs (not defaults) so weight comparison is fair.
PRICE_TREE_WEIGHT_PROBE = {
    "LGBMRegressor": {
        "learning_rate": 0.012, "n_estimators": 1000, "max_depth": 8,
        "num_leaves": 31, "min_child_samples": 50, "subsample": 0.7,
        "colsample_bytree": 0.7, "reg_alpha": 5.0, "reg_lambda": 5.0,
        "min_split_gain": 0.1, "objective": "mae", "metric": "mae",
    },
    "XGBRegressor": {
        "learning_rate": 0.02, "n_estimators": 1000, "max_depth": 8,
        "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 5.0, "reg_lambda": 5.0, "gamma": 0.3,
        "objective": "reg:absoluteerror", "eval_metric": "mae",
    },
    "CatBoostRegressor": {
        "learning_rate": 0.015, "iterations": 1000, "depth": 8,
        "l2_leaf_reg": 5.0, "subsample": 0.7, "rsm": 0.7,
        "min_child_samples": 50, "loss_function": "MAE", "eval_metric": "MAE",
        "verbose": 0,
    },
}

# Stage 2: hyperparam grid with weight fixed — more configs than before
PRICE_TREE_GRID = {
    "LGBMRegressor": [
        {"learning_rate": 0.008, "max_depth": 8, "reg_alpha": 5.0, "n_estimators": 1000},  # EP production
        {"learning_rate": 0.015, "max_depth": 6, "reg_alpha": 7.0, "n_estimators": 800},   # lower complexity
        {"learning_rate": 0.012, "max_depth": 8, "reg_alpha": 10.0, "n_estimators": 1000},  # higher reg
        {"learning_rate": 0.02, "max_depth": 10, "reg_alpha": 3.0, "n_estimators": 1200},   # higher capacity
        {"learning_rate": 0.008, "max_depth": 7, "reg_alpha": 5.0, "n_estimators": 800},    # conservative
        {"learning_rate": 0.01, "max_depth": 9, "reg_alpha": 8.0, "n_estimators": 1100},    # mid-range
        {"learning_rate": 0.025, "max_depth": 6, "reg_alpha": 5.0, "n_estimators": 900},    # fast/shallow
        {"learning_rate": 0.008, "max_depth": 8, "reg_alpha": 12.0, "n_estimators": 1000},  # very high reg
    ],
    "XGBRegressor": [
        {"learning_rate": 0.023, "max_depth": 8, "reg_alpha": 5.0, "n_estimators": 1000},  # EP production
        {"learning_rate": 0.015, "max_depth": 6, "reg_alpha": 8.0, "n_estimators": 800},
        {"learning_rate": 0.01, "max_depth": 8, "reg_alpha": 10.0, "n_estimators": 1000},
        {"learning_rate": 0.025, "max_depth": 10, "reg_alpha": 3.0, "n_estimators": 1200},
        {"learning_rate": 0.015, "max_depth": 7, "reg_alpha": 6.0, "n_estimators": 900},
        {"learning_rate": 0.02, "max_depth": 9, "reg_alpha": 8.0, "n_estimators": 1100},
        {"learning_rate": 0.01, "max_depth": 6, "reg_alpha": 12.0, "n_estimators": 800},
        {"learning_rate": 0.03, "max_depth": 8, "reg_alpha": 4.0, "n_estimators": 1000},
    ],
    "CatBoostRegressor": [
        {"learning_rate": 0.015, "depth": 8, "l2_leaf_reg": 5.0, "iterations": 1000},  # EP production
        {"learning_rate": 0.01, "depth": 9, "l2_leaf_reg": 7.0, "iterations": 800},
        {"learning_rate": 0.02, "depth": 7, "l2_leaf_reg": 5.0, "iterations": 1200},
        {"learning_rate": 0.012, "depth": 8, "l2_leaf_reg": 8.0, "iterations": 1000},
        {"learning_rate": 0.018, "depth": 6, "l2_leaf_reg": 4.0, "iterations": 900},
    ],  # CatBoost is 2-3× slower — 5 configs is enough
}
# Total tree trials: 12 (stage 1) + 8+8+5 (stage 2) = 33 trials

# ── Linear model grids ───────────────────────────────────────────
LINEAR_ALPHA_GRID = {
    "Ridge": np.logspace(-3, 1, 15).tolist(),       # 0.001 to 10
    "Lasso": np.logspace(-3, 0, 12).tolist(),        # 0.001 to 1
    "ElasticNet": np.logspace(-3, 0, 10).tolist(),   # 0.001 to 1
}
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
# Total linear trials per feature set:
#   Ridge:      15 alpha × 24 preprocessing = 360
#   Lasso:      12 alpha × 24 preprocessing = 288
#   ElasticNet: 10 alpha × 5 l1_ratio × 24 preprocessing = 1200
#   Total:      1848 (~seconds each, so a few minutes wall-clock)

# ── Preprocessing ─────────────────────────────────────────────────
TARGET_TRANSFORMS = ["none", "log_shift", "yeo_johnson"]
FEATURE_SCALERS = ["standard", "robust"]  # "none" omitted — linear models require scaling
WEIGHT_HALF_LIVES = [None, 365, 730, 1095]

# Tree models: no scaler, no target transform (invariant to monotonic transforms).
# Only weight_half_life is searched.
TREE_PREPROCESSING = {"scaler": "none", "target_transform": "none"}

# Linear models: full preprocessing grid.
LINEAR_PREPROCESSING_GRID = list(itertools.product(
    TARGET_TRANSFORMS, FEATURE_SCALERS, WEIGHT_HALF_LIVES,
))  # 3 × 2 × 4 = 24 combos

# ── Gen/load dataset-level suggestions ────────────────────────────
def suggest_dataset_params(trial) -> dict:
    return {
        "log_target": trial.suggest_categorical("log_target", [True, False]),
        "lags_target": trial.suggest_categorical("lags_target", [None, 1, 6, 12]),
        "scaler": trial.suggest_categorical("scaler", ["standard", "robust", "minmax"]),
    }
```

---

## 5a.2 MLflow Helpers

**`energy_forecasting/modeling/mlflow_utils.py`** — new file.

```python
class TrackedRun:
    """Context manager for MLflow runs with tag validation.

    Validates required tags at __enter__ — fails BEFORE any training compute.
    Auto-sets tags that can be derived from arguments (dataset_name, cv_mode, etc.).
    """
    def __init__(self, experiment: str, *, dataset_name: str = None, **tags):
        ...

    def __enter__(self):
        # 1. Validate experiment exists
        # 2. Validate all required tags provided
        # 3. Auto-set: dataset_name (from arg), timestamp, python_version
        # 4. Start MLflow run, set tags
        # 5. Return run context
        ...
```

**Auto-tagging:** These tags are set automatically from function arguments, never manually:
- `dataset_name` — from the dataset name passed to training (provenance tracked via `mlflow.log_input()`)
- `cv_mode` — from `TimeSeriesSplitter.mode` ("expanding"/"sliding")
- `cv_folds` — from `TimeSeriesSplitter.n_splits`
- `holdout_days` — from the holdout parameter
- `n_features` — from `X.shape[1]`
- `n_train_rows` — from training data size
- `model_class` — from `type(model).__name__`

**Manual tags** (must be provided):
- `stage` — "feature_selection", "model_training", "production"
- `feature_version` — which feature list/selection was used

Other helpers: `audit_experiment`, `archive_runs`, `get_best_run`, `compare_models`, `compare_feature_sets`, `export_experiment_summary`.

---

## 5a.3 Metrics

**`energy_forecasting/modeling/metrics.py`**

```python
def calculate_metrics(y_true, y_pred, y_baseline=None) -> dict:
    """RMSE, MAE, ME, R², MAPE, sMAPE. + skill scores if baseline provided."""

def calculate_pi_metrics(y_true, y_lower, y_upper) -> dict:
    """PI coverage, mean width, median width."""

def calculate_peak_metrics(y_true, y_pred, peak_hours=PEAK_HOURS) -> dict:
    """Metrics restricted to peak hours."""
```

---

## 5a.4 Cross-Validation & Holdout

**`energy_forecasting/modeling/cv.py`** — new file.

```python
class TimeSeriesSplitter:
    """Day-boundary-aware time series CV.

    All splits enforce full-day boundaries:
    - Train ends at hour 23
    - Test starts at hour 0
    - Test/train sizes rounded to whole days

    Holdout is carved out BEFORE CV — CV never sees holdout data.
    """
    def __init__(
        self,
        n_splits: int,                     # caller must pass SEARCH_CV_FOLDS or VALIDATION_CV_FOLDS
        test_days: int | None = None,
        mode: str = "expanding",       # "expanding" or "sliding"
        gap_days: int = 0,
        step_days: int | None = None,  # None = non-overlapping
    ): ...

    def split(self, index: pd.DatetimeIndex) -> Iterator[tuple[ndarray, ndarray]]: ...
```

### CV & Holdout Diagram

```
Full dataset (e.g. 2015-01-01 to 2026-03-01, ~98K hours)
├──────────────────────────────────────────────┤
│                                              │

Step 1: Carve out holdout (last 90 days)
├───────────── Train+CV pool ──────────────┤── Holdout (90d) ──┤
│  CV operates only in this portion        │  Sacred — final    │
│                                          │  evaluation only   │

Step 2: CV operates on Train+CV pool only
Mode: "expanding" (train grows), 5 folds, non-overlapping

Fold 1: ├─── Train ────────────────────────┤── Test ──┤
Fold 2: ├─── Train ───────────────┤── Test ──┤
Fold 3: ├─── Train ──────┤── Test ──┤
Fold 4: ├─── Train ─┤── Test ──┤
Fold 5: ├─ Train ┤── Test ──┤

         ← older                    newer →

All boundaries fall on day boundaries (23:00 → 00:00).
Test folds are equal-sized and move backward.
Train always starts at the beginning of the pool.

Mode: "sliding" — train is fixed-size window instead of growing.

Step 3: Final model trained on full Train+CV pool
Step 4: Evaluated on holdout
Step 5: MAPIE conformal calibration uses internal CV within final training
```

---

## 5a.5 Training Loop

**`energy_forecasting/modeling/training.py`**

### Pipeline Construction

```python
def build_pipeline(
    model,
    scaler: str = "standard",
    target_transform: str = "none",
) -> Pipeline:
    """Build sklearn Pipeline: scaler → TransformedTargetRegressor(model).

    Scaler options: "standard" (StandardScaler), "robust" (RobustScaler), "none".
    Target transforms: "none", "log_shift", "yeo_johnson".
    TransformedTargetRegressor auto-inverts on predict().
    """
```

### Core Training Function

```python
def train_model(
    dataset_path: Path,                 # Parquet file from prepare_dataset()
    model,                              # sklearn-compatible regressor
    experiment: str,
    tags: dict,                         # must include "stage", "feature_version"
    scaler: str = "standard",
    target_transform: str = "none",
    weight_half_life: float | None = None,
    cv: TimeSeriesSplitter | None = None,
    holdout_days: int = HOLDOUT_DAYS,
) -> str:  # returns run_id
# dataset_name derived from dataset_path.stem (e.g. "price_slim_v1")
```

Steps:
1. **Load dataset** from Parquet via `load_dataset(dataset_path)`. Get X, y.
2. **MLflow validation:** Open `TrackedRun` — validates tags + experiment **before training**. Register dataset via `log_dataset_to_run(X, dataset_path)`.
3. **Holdout split:** Last `holdout_days × 24` rows. Day-boundary aligned.
4. **Build pipeline:** `build_pipeline(model, scaler, target_transform)`.
5. **Sample weights:** If `weight_half_life` set, compute from `day_index`.
6. **CV evaluation:** Clone pipeline per fold, compute metrics on train portion.
7. **MAPIE wrapping:** `CrossConformalRegressor(pipeline, confidence_level=PI_CONFIDENCE_LEVEL, cv=PI_CV_FOLDS)`.
8. **Final fit:** `fit_conformalize(X_train, y_train)`.
9. **Holdout evaluation:** `predict_interval(X_holdout)` → metrics + PI metrics.
10. **Log everything** to MLflow: metrics, params, model artifact, dataset reference.
11. Return `run_id`.

### Sample Weights

```python
def compute_sample_weights(day_index: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential decay. w = exp(ln(2)/half_life × (t - t_max))."""
```

---

## 5a.6 Prediction Intervals

**`energy_forecasting/modeling/intervals.py`**

### Base Model Intervals (MAPIE)

```python
def wrap_with_intervals(pipeline, confidence_level=PI_CONFIDENCE_LEVEL, cv=PI_CV_FOLDS):
    """Wrap sklearn pipeline with CrossConformalRegressor."""

def predict_with_intervals(model, X) -> tuple[ndarray, ndarray, ndarray]:
    """Returns (point_pred, lower, upper)."""
```

### Ensemble Intervals (Post-Hoc Conformal)

Method-agnostic — works for any ensemble (blend, stacking, greedy, etc.):

```python
def calibrate_ensemble_intervals(y_holdout, y_holdout_pred, confidence_level):
    """Compute conformal quantile from holdout residuals."""

def predict_ensemble_intervals(y_pred, conformal_quantile):
    """Apply: y_pred ± quantile."""
```

### XGBoost eps fix

EMA needed `AbsoluteConformityScore.eps = 1e-4` for XGBoost float32 precision in MAPIE 0.9.x. Verify whether MAPIE 1.3 `CrossConformalRegressor` still needs this. Test empirically during implementation.

---

## 5a.7 Dataset Management

**`energy_forecasting/modeling/datasets.py`** — new file.

Separates feature computation from training. Datasets are Parquet files on disk, tracked in MLflow via `mlflow.log_input()` — not as runs or experiments (see `docs/mlflow_conventions.md`).

```python
DATASET_DIR = Path("data/processed/datasets")

def prepare_dataset(
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    name: str,                          # e.g. "price_slim_v1"
) -> Path:
    """Compute features via engineer_features(), save to
    data/processed/datasets/{name}.parquet. Returns file path.
    No MLflow logging here — that happens in train_model().
    """

def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load X, y from Parquet file."""

def update_dataset(
    path: Path,
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
) -> Path:
    """Extend existing dataset with new rows via extend_features().
    Overwrites the Parquet file in place. Returns same path.
    """

def find_dataset(name: str) -> Path | None:
    """Check if data/processed/datasets/{name}.parquet exists."""

def log_dataset_to_run(X: pd.DataFrame, path: Path):
    """Called inside a training run to register the dataset with MLflow.
    Name derived from path.stem.
    dataset = mlflow.data.from_pandas(X, source=str(path), name=path.stem)
    mlflow.log_input(dataset, context="training")
    """
```

Workflow:
1. `find_dataset("price_slim_v1")` — check if Parquet exists.
2. If found and up-to-date: use it. If stale: `update_dataset()`. If missing: `prepare_dataset()`.
3. Training functions call `load_dataset(path)` for data, then `log_dataset_to_run()` inside the MLflow run to record provenance.

---

## 5a.8 Baselines

**`energy_forecasting/modeling/baselines.py`**

```python
def naive_lag(y, lag=24) -> pd.Series:       # same hour yesterday
def naive_weekly(y) -> pd.Series:            # same hour last week
def rolling_day_ahead(y, holdout_days=HOLDOUT_DAYS) -> tuple:  # SARIMA 24h-ahead
```

---

## 5a.9 Forecasting Functions

**`energy_forecasting/modeling/forecasting.py`** — new file (not `recursive.py` — that name was misleading since direct prediction is the default and this handles all gen/load forecasting modes).

```python
def forecast_direct(model, X_test: pd.DataFrame) -> pd.DataFrame:
    """Direct prediction — no lag updates. Default mode.
    Returns DataFrame: fitted, lower, upper.
    """

def forecast_with_lags(
    model, X_test, y_train, lag_columns, target_name,
) -> pd.DataFrame:
    """Recursive one-step-ahead with lag feature updates.
    Only activated when lags_target is not None (rare in practice).
    """
```

---

# Stage 5b: Gen/Load Models

## EMA Reference Accuracy (DE, XGBoost only, 7-day horizon)

From EMA's output on German TSO regions. These are per-fold numbers (each fold is one 7-day horizon), so they vary. PI coverage was poor in EMA (20-50% for most targets) — improving this is an explicit goal.

| Target | Region | Typical MAE | Typical RMSE | R² range | Notes |
|---|---|---|---|---|---|
| wind_onshore | 50Hz | 560-1120 MW | 750-1630 MW | 0.70-0.87 | Best on calm weeks |
| wind_onshore | Amprion | 585-1130 MW | 790-1410 MW | 0.57-0.91 | |
| wind_onshore | TenneT | 795-1850 MW | 1010-2440 MW | 0.78-0.84 | Largest region, highest absolute error |
| wind_onshore | TransnetBW | 145-200 MW | 175-260 MW | -0.46-0.87 | Smallest, high variance in R² |
| wind_offshore | 50Hz | 130-230 MW | 175-310 MW | 0.55-0.93 | |
| wind_offshore | TenneT | 410-970 MW | 540-1300 MW | 0.37-0.92 | High variance |
| solar | 50Hz | 350-810 MW | 660-1470 MW | 0.77-0.96 | Winter metrics worse |
| solar | Amprion | 300-500 MW | 650-1150 MW | 0.77-0.94 | |
| solar | TenneT | 465-1160 MW | 950-2110 MW | 0.73-0.97 | |
| solar | TransnetBW | 190-390 MW | 365-695 MW | 0.86-0.97 | |
| load | 50Hz | 590-820 MW | 730-1010 MW | 0.62-0.80 | |
| load | Amprion | 555-1100 MW | 685-1350 MW | 0.77-0.94 | |
| load | TenneT | 660-1100 MW | 830-1480 MW | 0.69-0.88 | |
| load | TransnetBW | 425-730 MW | 485-1080 MW | 0.34-0.85 | PI coverage only 7-20% |

These are single-model (XGBoost) numbers. EMA's stacking ensemble typically improves by 10-30% over the best single model.

## 5b.1 Gen/Load Features

Features come from three sources, all searched jointly via Optuna:

1. **Weather FE** (target-specific class): wind speed/direction aggregation, air density, irradiance derivatives, temperature interactions. Optuna searches over spatial aggregation method (mean, IDW, capacity-weighted), lag sets ("small"=[1,6] vs "large"=[1,6,12,24]), and computed variable subsets.
2. **Temporal**: hour (cyclical sin/cos), day of week (cyclical), holiday indicator. From `GEN_LOAD_FEATURES` in `config/features.py` (22 features).
3. **Dataset params** (via `suggest_dataset_params`): `log_target` (True/False), `lags_target` (None/1/6/12), `scaler` (standard/robust/minmax).

The weather FE class selection and config is the main differentiator between targets. The temporal and dataset params are shared.

## 5b.2 Gen/Load Training Pipeline

**`energy_forecasting/modeling/gen_load.py`**

```python
def train_gen_load_model(
    target: str,                    # e.g. "wind_onshore"
    region: str,                    # e.g. "DE_50HZ"
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame | None,
    model_type: str = "LGBMRegressor",
    optuna_trials: int = 70,
    cv_folds: int = SEARCH_CV_FOLDS,
) -> str:
```

Steps:
1. Weather FE via appropriate class (`WeatherWindPowerFE`/`WeatherSolarPowerFE`/`WeatherLoadFE`).
2. Temporal features, optional lagged targets (default: None → direct prediction).
3. Optuna with `TPESampler` — joint search over model hyperparams + dataset params + weather FE config. Uses `SEARCH_CV_FOLDS` (3) and `MedianPruner`. Tree models use `early_stopping_rounds=50`.
4. Train with best params via `train_model()` (final evaluation with `VALIDATION_CV_FOLDS`).
5. Save weather FE output to `data/processed/weather_features/{target}_{region}.parquet` for load→price transfer.

### Training order

Wind/solar models must train before load if load uses wind/solar forecasts as exogenous features (EMA pattern). Order: wind_onshore → wind_offshore → solar → load. Within each target, regions are independent and can run in parallel.

## 5b.3 Gen/Load Ensembling

Per target/region: train 3 base models (LightGBM, XGBoost, ElasticNet), each with its own Optuna search. Ensemble via stacking with Ridge meta-learner (EMA's existing pattern):

```python
def ensemble_gen_load(
    target: str,
    region: str,
    base_run_ids: list[str],       # 3 base model MLflow run IDs
) -> str:
    """Stacking ensemble for gen/load.

    1. Load OOF predictions from each base model (logged as artifacts).
    2. Stack as meta-features: [pred_model1, pred_model2, pred_model3].
       Optionally include PI bounds as additional meta-features.
    3. Train Ridge(positive=True) meta-learner on OOF predictions.
    4. Evaluate on holdout.
    5. Post-hoc conformal calibration for ensemble PI.
    6. Log to generation/{target} MLflow experiment.
    """
```

If the ensemble doesn't beat the best single model on holdout, fall back to the single model. This is checked automatically.

## 5b.3.1 Historical Forecasts via OOF Reuse

EP's price model (Stage 5c) needs leak-free historical forecasts of gen/load
values. EMA solves this with a separate 1139-line batch pipeline
(`generate_historical_forecasts.py`) that re-runs trained models at
hundreds of historical cutoffs in `backtest` or `hindcast` mode.

In the merged repo we collapse that pipeline into the final-model training
pass:

1. **218 sliding-window weekly CV folds** in the final pass
   (`GEN_LOAD_HISTORICAL_FOLDS`). Each fold is a leak-free 168h forecast at
   that cutoff. 218 × 168h ≈ 4.18 years of OOF coverage spanning
   2022-01-15 → 2026-03-27, matching the `hist_forecast` weather window.
   (Originally 40 folds / ~9.5 months; bumped during Phase A 2026-05-07.)
2. **Per-fold weather swap.** Test rows are evaluated on `hist_forecast`
   weather (Open-Meteo's "what would have been forecast as of this hour")
   instead of the actual weather archive. This gives the OOFs realistic
   forecast errors — equivalent to EMA's `mode="backtest"`. Implemented
   via `train_model(..., test_dataset_path=...)`: the gen/load training
   builds two parallel feature matrices (one per weather source), CV uses
   the actual one for training rows and the hist_forecast one for test
   rows. Training rows always come from actual weather (matches EMA's
   `df_hist`).
3. **Concatenate OOF + holdout** on the time index and save to
   `data/processed/historical_forecasts/{target}_{region}.parquet` with
   schema `[y_true, y_pred, y_lower, y_upper]`. PI bounds are NaN when the
   model uses recursive lag features (MAPIE bypassed for those — same as
   EMA).
4. **National aggregates.** For per-TSO targets we additionally save
   `{target}_DE_NATIONAL.parquet` summed across regions on the common
   index, matching EMA's `export_national_forecasts.py` collapse step.

What we give up vs EMA's standalone pipeline:
- **Pre-2022 hindcast coverage.** Stage 5c falls back to SMARD's
  `prognostizierte_*` for pre-window timestamps if needed.

What we keep:
- Realistic forecast errors via per-fold weather swap.
- Tight integration with the training run — no parallel codebase.
- Same OOF schema EP's `build_ema_training_data` consumes.

## 5b.4 Gen/Load Baselines

7-day baselines (different from price — 24h lag doesn't apply):

```python
def naive_persistence_7d(y) -> pd.Series:
    """Same hour, same day-of-week, previous week. Lag = 168h."""

def naive_seasonal_7d(y) -> pd.Series:
    """Same hour, same day-of-week, average of last 4 weeks."""

def climatological_baseline(y, window_days=90) -> pd.Series:
    """Hour-of-day × day-of-week mean over trailing window."""
```

Skill scores in gate 5b are computed against `naive_persistence_7d`.

---

# Stage 5c: Price Models

## 5c.0 Feature Infrastructure Prerequisites

Stage 5c needs several additions to the feature engineering layer before `PRICE_FEATURES_MAX` can be materialised. These are strictly additive — no existing features change.

### Gen/load forecast integration

Price models benefit from knowing what tomorrow's generation/load will look like. ENTSO-E prognosis data (`prog_*` features) is one source, but our own 5b gen/load models produce more accurate forecasts. Gen/load forecasts enter the price feature pipeline as **derived columns** loaded directly from on-disk parquet artifacts produced by 5b — no MLflow lookup, no separate cache layer.

**Architecture:**

1. **During 5b training**, each gen/load model writes leak-free predictions to disk at `data/processed/historical_forecasts/{target}_{region}.parquet` (per-TSO) and `data/processed/historical_forecasts/{target}_DE_NATIONAL.parquet` (sum across regions on the common index). Schema: `[y_true, y_pred, y_lower, y_upper]`. Training-period rows come from OOF predictions (CV folds where each row's prediction comes from a model that never saw it during fitting); holdout-period rows come from the final model evaluated on held-out days. `y_lower`/`y_upper` are NaN for upstream base models that use recursive lags and bypass MAPIE — expected, since the price model only consumes `y_pred`.

2. **During price dataset preparation**, a new function `load_gen_load_forecasts(index, regions=None)` reads these parquet files directly from disk:
   - Defaults to national aggregates (`{target}_DE_NATIONAL.parquet`); per-region available via the `regions` argument.
   - Joins on the requested timestamp index, returns a DataFrame with `_derived_forecast_*` columns.
   - Computes `forecast_residual = forecast_load - forecast_wind_on - forecast_wind_off - forecast_solar` in the loader.
   - Treats NaN PI bounds as missing PI, not missing predictions — does not drop rows on `y_lower`/`y_upper` NaN.

3. **`_prepare_working_df()`** gets a new branch: if any `_derived_forecast_*` columns are needed, call `load_gen_load_forecasts()` and merge the returned DataFrame.

4. **At inference time**, the gen/load models run live against the current day's inputs and write into the same `historical_forecasts/` location for that day's slice; the price pipeline reads from the same files via the same loader.

**Availability:** Gen/load forecasts are day-ahead predictions (like TSO prognosis). Availability rule uses **offset=0** — available for the target day with no lag. Leakage protection comes from OOF predictions written by 5b during training, not from lagging at price-prep time.

**SHORT_NAMES additions:**

```python
"forecast_wind_on":       "_derived_forecast_wind_on",
"forecast_wind_off":      "_derived_forecast_wind_off",
"forecast_solar":         "_derived_forecast_solar",
"forecast_load":          "_derived_forecast_load",
"forecast_gen_load_diff": "_derived_forecast_gen_load_diff",
"forecast_residual":      "_derived_forecast_residual",
```

**Availability rule addition:**

```python
AvailabilityRule(
    pattern="forecast_*",
    max_offset_days=0,
    cutoff_hour=None,
    reason="Day-ahead forecasts from gen/load models. "
           "Training-period rows are OOF predictions (no leakage); "
           "holdout-period rows are final-model predictions; "
           "at inference, models produce live forecasts.",
),
```

**Implementation constraints:**

- **Pre-flight check**: `load_gen_load_forecasts()` raises a clear error listing missing files if any expected `data/processed/historical_forecasts/{target}_{region}.parquet` is absent. No silent fallback to no-forecast.
- **Source of truth**: The parquet files *are* the canonical artifact. No MLflow run-id lookup, no `production=true` tag step, no separate cache layer — the on-disk parquet IS the cache. If 5b is retrained, the files are overwritten in place; subsequent 5c retraining picks up the fresh forecasts automatically.
- **Development against missing 5b artifacts**: If 5b parquet files are missing, the loader fails with a clear error naming the gap. Once 5b has produced them, the price pipeline runs without further setup.

**Training order implication:** Price model training has a hard file dependency on `data/processed/historical_forecasts/`. The Gate 5b → 5c "Before 5c" pre-flight verifies all expected files exist before any price dataset including forecast features can be built.

**Relationship to weather features:** The earlier plan asked whether weather features should enter price models directly. **Decision:** defer direct weather features. Gen/load forecasts capture the weather→generation relationship more compactly and avoid the per-TSO aggregation problem. If ensemble MAE is still gapping vs. EP production after 5c, revisit this by adding top-signal weather features (e.g. national irradiance, wind speed) to `PRICE_FEATURES_MAX`.

### EEG regime and negative-price features

Three new derived columns, with different availability treatment.

**`eeg_regime`** — deterministic categorical (0/1/2/3) based on `EEG_REGIME_DATES` in `config/modeling.py`. Bare name is fine because it's purely date-driven.

- **SHORT_NAMES**: `"eeg_regime": "_derived_eeg_regime"`
- **Availability rule**: offset=0 (deterministic, like `is_holiday`)
- **Computation**: New branch in `_prepare_working_df()` — maps each timestamp's date to a regime integer using the threshold dates. A small utility `compute_eeg_regime(index)` in `features/market.py`.

**`neg_price_frac_30d`, `neg_price_frac_90d`, `neg_price_depth_30d`** — rolling stats on the target price. These must use the `_d1` suffix so the engine lags them by one day; bare names would fail the validator.

- **SHORT_NAMES**:
  ```python
  "neg_price_frac_30d":  "_derived_neg_price_frac_30d",
  "neg_price_frac_90d":  "_derived_neg_price_frac_90d",
  "neg_price_depth_30d": "_derived_neg_price_depth_30d",
  ```
- **Availability rule**: `neg_price_*` → offset=-1 (same as price)
- **Computation**: New branch in `_prepare_working_df()` — computes naive rolling 30d/90d stats on the full `price` column (no lag shift; the engine will do the D-1 lag via the `_d1` suffix).

**Why `_d1` works cleanly:** The engine parses `neg_price_frac_30d_d1` as an Aggregation with `end_day=-1`. On a pre-computed column, a single-day D-1 aggregation simply returns yesterday's value. The rolling stat is computed once at dataset prep; the engine picks up the correct lagged value per row.

### Extending `compute_generation_pct()` for per-technology pct_prog_*

SMARD provides `prognostizierte_erzeugung_onshore`, `prognostizierte_erzeugung_offshore`, and `prognostizierte_erzeugung_photovoltaik`, already mapped to `prog_gen_wind_on`, `prog_gen_wind_off`, `prog_gen_solar` in `config/columns.py`. But `compute_generation_pct()` in `features/market.py` only emits `_derived_pct_prog_other` and `_derived_pct_prog_wind_pv`. Three more derived columns are needed:

```python
# In compute_generation_pct(), extend the add_prognosticated_pct branch:
prog_wind_on  = "prognostizierte_erzeugung_onshore"
prog_wind_off = "prognostizierte_erzeugung_offshore"
prog_solar    = "prognostizierte_erzeugung_photovoltaik"
if prog_wind_on in df.columns:
    result["_derived_pct_prog_wind_on"]  = df[prog_wind_on] / pt
if prog_wind_off in df.columns:
    result["_derived_pct_prog_wind_off"] = df[prog_wind_off] / pt
if prog_solar in df.columns:
    result["_derived_pct_prog_solar"]    = df[prog_solar] / pt
```

**SHORT_NAMES additions:**

```python
"pct_prog_solar":    "_derived_pct_prog_solar",
"pct_prog_wind_on":  "_derived_pct_prog_wind_on",
"pct_prog_wind_off": "_derived_pct_prog_wind_off",
```

The existing `pct_prog_*` availability rule (offset=0) already covers the new names.

### Summary of `config/availability.py` changes

Two new rules to add (order matters — more specific patterns before more general ones):

```python
AvailabilityRule("forecast_*", 0, None, "Day-ahead forecasts from gen/load models."),
AvailabilityRule("neg_price_*", -1, None, "Rolling stats on target price."),
AvailabilityRule("eeg_regime", 0, None, "Deterministic regime indicator from date."),
```

### Summary of `config/columns.py` SHORT_NAMES additions

Ten new entries total:
- 6 forecast features (`forecast_wind_on`, `forecast_wind_off`, `forecast_solar`, `forecast_load`, `forecast_gen_load_diff`, `forecast_residual`)
- 3 per-technology prognosis percentages (`pct_prog_solar`, `pct_prog_wind_on`, `pct_prog_wind_off`)
- 1 EEG regime indicator (`eeg_regime`)
- 3 negative-price rolling stats (`neg_price_frac_30d`, `neg_price_frac_90d`, `neg_price_depth_30d`)

That's 13 entries, not 10 — listed above under their respective subsections.

### Summary of `features/engine.py` (`_prepare_working_df`) changes

Three new branches to add:

1. `_derived_forecast_*` → call `load_gen_load_forecasts(index)` and merge
2. `_derived_eeg_regime` → call `compute_eeg_regime(index)`
3. `_derived_neg_price_*` → compute naive rolling stats on the `price` column

The existing `_derived_pct_prog_*` branch already handles the three new percentages automatically once `compute_generation_pct()` is extended.

---

## 5c.1 PRICE_FEATURES_MAX

**`energy_forecasting/config/features.py`** — add `PRICE_FEATURES_MAX` (~350 features).

Everything plausible. The feature selection pipeline (§5c.2) will reduce this to an optimal subset. Categories:

```python
PRICE_FEATURES_MAX: list[str] = [
    *PRICE_FEATURES_FULL,  # 130 base features

    # ── All neighbour price EWMAs (11 countries × 3 spans = 33) ──────
    # FR, NL, AT, CH already in FULL. Add: DK1, DK2, BE, PL, CZ, NO2, SE4
    "price_dk1_ewma_6_d1", "price_dk1_ewma_24_d1", "price_dk1_ewma_2160_d1",
    "price_dk2_ewma_6_d1", "price_dk2_ewma_24_d1", "price_dk2_ewma_2160_d1",
    "price_be_ewma_6_d1",  "price_be_ewma_24_d1",  "price_be_ewma_2160_d1",
    "price_pl_ewma_6_d1",  "price_pl_ewma_24_d1",  "price_pl_ewma_2160_d1",
    "price_cz_ewma_6_d1",  "price_cz_ewma_24_d1",  "price_cz_ewma_2160_d1",
    "price_no2_ewma_6_d1", "price_no2_ewma_24_d1", "price_no2_ewma_2160_d1",
    "price_se4_ewma_6_d1", "price_se4_ewma_24_d1", "price_se4_ewma_2160_d1",

    # ── All neighbour price hourly lags h24, h48 (11 countries × 2) ──
    "price_nl_h24", "price_nl_h48",
    "price_at_h24", "price_at_h48",
    "price_dk1_h24", "price_dk1_h48",
    "price_dk2_h24", "price_dk2_h48",
    "price_be_h24", "price_be_h48",
    "price_pl_h24", "price_pl_h48",
    "price_cz_h24", "price_cz_h48",
    "price_no2_h24", "price_no2_h48",
    "price_se4_h24", "price_se4_h48",
    # FR h48, CH h48 already in FULL

    # ── Additional price EWMA spans ──────────────────────────────────
    "price_ewma_48_d1",      # 2-day EWMA
    "price_ewma_336_d1",     # 14-day EWMA
    "price_ewma_720_d1",     # 30-day EWMA

    # ── Additional price lags ────────────────────────────────────────
    "price_h72",             # D-3
    "price_h96",             # D-4

    # ── Additional price rolling windows ─────────────────────────────
    "price_d14_d1_max", "price_d14_d1_min", "price_d14_d1_range",
    "price_d3_d1_avg",       # 3-day mean
    "price_d5_d1_avg",       # 5-day mean
    "price_d5_d1_std",
    "price_d7_d1_h0_h8_avg",     # off-peak hours
    "price_d7_d1_h20_h24_avg",   # evening hours

    # ── All generation type D-2 daily means ──────────────────────────
    # Already have: wind_on, wind_off, solar, gas, load. Add rest:
    "gen_biomass_d2", "gen_nuclear_d2", "gen_pumped_d2",
    "gen_other_conv_d2",

    # ── All generation type h48 lags ─────────────────────────────────
    # Already have: wind_on, wind_off, solar. Add rest:
    "gen_gas_h48", "gen_lignite_h48", "gen_coal_h48",
    "gen_nuclear_h48", "gen_biomass_h48", "gen_hydro_h48",
    "gen_pumped_h48", "gen_other_h48", "gen_other_renew_h48",
    "load_h48",

    # ── All generation type morning actuals D-1 eh7 ──────────────────
    # Already have some. Add remaining:
    "gen_wind_off_d1_eh7", "gen_biomass_d1_eh7",
    "gen_pumped_d1_eh7", "gen_hydro_d1_eh7",
    "gen_other_d1_eh7", "gen_other_renew_d1_eh7",
    "gen_coal_d1_eh7",

    # ── All generation pct D-2 ───────────────────────────────────────
    # Already have ~6. Add:
    "gen_pct_biomass_d2", "gen_pct_pumped_d2",
    "gen_pct_other_renew_d2", "gen_pct_other_conv_d2",

    # ── Cross-border per-country net flows D-2 ───────────────────────
    "net_export_austria_d2", "net_export_belgium_d2",
    "net_export_czech_republic_d2", "net_export_denmark_1_d2",
    "net_export_denmark_2_d2", "net_export_france_d2",
    "net_export_netherlands_d2", "net_export_norway_2_d2",
    "net_export_poland_d2", "net_export_sweden_4_d2",
    "net_export_switzerland_d2",

    # ── Additional commodity rolling stats ────────────────────────────
    "ttf_d14_d2_avg", "ttf_d30_d2_avg", "ttf_d30_d2_std",
    "brent_d14_d2_avg", "brent_d30_d2_avg",
    "carbon_d14_d2_avg", "carbon_d30_d2_avg", "carbon_d30_d2_std",
    "ttf_d7_d2_std", "brent_d7_d2_std", "carbon_d7_d2_std",

    # ── Additional EWMA ──────────────────────────────────────────────
    "residual_load_ewma_6_d1_h7",
    "gen_wind_off_ewma_24_d1_h7", "gen_wind_off_ewma_168_d1_h7",
    "load_ewma_24_d1_h7", "load_ewma_168_d1_h7",
    "gen_gas_ewma_24_d2",

    # ── Forecast daily aggregate variants ────────────────────────────
    "prog_gen_total_daily_max", "prog_gen_total_daily_avg",
    "prog_load_daily_max", "prog_load_daily_avg",
    "prog_residual_daily_max", "prog_residual_daily_avg",
    "prog_gen_solar_daily_max", "prog_gen_solar_daily_sum",
    "prog_gen_wind_on_daily_max",

    # ── Prognosticated percentages ───────────────────────────────────
    # Requires extending compute_generation_pct() to emit per-technology
    # derived columns and adding three new SHORT_NAMES entries. See §5c.0
    # "Missing SHORT_NAMES and derived columns".
    "pct_prog_solar", "pct_prog_wind_on", "pct_prog_wind_off",

    # ── Price volatility / momentum ──────────────────────────────────
    "price_d2_d1_std__x__price_d7_d1_std",  # short/long vol ratio
    "price_d7_d1_avg__x__price_d30_d1_avg", # short/long momentum

    # ── EEG regime indicators ────────────────────────────────────────
    # `eeg_regime` is deterministic (date-based), so no suffix needed.
    # `neg_price_*` are rolling stats on the target price, so they need `_d1`
    # to make the engine pick up yesterday's value of the pre-computed column.
    "eeg_regime",                # categorical: 0/1/2/3 for pre-2023/4h/2h/solarspitzengesetz
    "neg_price_frac_30d_d1",     # fraction of hours with price < 0 in last 30 days
    "neg_price_frac_90d_d1",     # same, 90 days
    "neg_price_depth_30d_d1",    # mean price during negative hours, last 30 days

    # ── Gen/load forecast features (from 5b models) ──────────────────
    # Day-ahead forecasts from our own gen/load models. At training time
    # these come from OOF predictions logged to MLflow in 5b; at inference
    # time they come from running the gen/load models first. Availability
    # offset=0 (like TSO prognosis). See §5c.0 for the integration architecture.
    "forecast_wind_on",          # national aggregate wind onshore forecast
    "forecast_wind_off",         # national aggregate wind offshore forecast
    "forecast_solar",            # national aggregate solar forecast
    "forecast_load",             # national aggregate load forecast
    "forecast_gen_load_diff",    # national gen-load difference forecast
    "forecast_residual",         # derived: forecast_load - forecast_wind_on - forecast_wind_off - forecast_solar

    # ── Additional interaction terms ─────────────────────────────────
    "price_ewma_24_d1__x__hour_sin",
    "prog_residual__x__is_weekend",
    "prog_gen_solar__x__hour_sin",
    "prog_gen_wind_pv__x__prog_load",
    "price_d7_d1_avg__x__day_index",
    "carbon_ewma_24_d2__x__gen_pct_gas_d2",
    "ttf_ewma_24_d2__x__gen_pct_gas_d2",
    "prog_residual__x__is_holiday",
    "price_d2_d1_std__x__day_index",
    "load_d1_eh7__x__day_index",
]
```

This puts us at ~350 features. Some will require new SHORT_NAMES entries and availability rules.

---

## 5c.2 Feature Selection Pipeline

**`energy_forecasting/modeling/feature_selection.py`** — new file.

Feature selection is an **MLflow experiment** (`price/feature_selection`). Each run produces a candidate feature set with metrics. Subsequent training experiments reference these.

```python
def correlation_filter(X, y, min_target_corr=0.02, max_pair_corr=0.9999) -> list[str]:
    """Remove near-zero-correlation and near-duplicate features."""

def shap_importance(model, X_val, feature_names) -> pd.Series:
    """Mean |SHAP value| via TreeExplainer. Returns sorted Series."""

def shap_cutoff_search(X_train, y_train, X_val, y_val, shap_ranking) -> list[int]:
    """Coarse then fine grid over top-N features.
    Returns list of N values that are local minima (not just the single best).
    """

def rfecv_select(X, y, initial_features, model, cv, step=3, min_features=10) -> tuple[list[str], dict]:
    """sklearn RFECV with TimeSeriesSplitter.
    Returns (selected_features, rfecv_curve_dict) — curve includes all N values tested.
    """

def run_feature_selection(
    dataset_path: Path,
    model_type: str = "LGBMRegressor",
) -> list[str]:
    """Full pipeline: correlation → SHAP → RFECV.

    Produces MULTIPLE candidate feature sets (not just one winner):
    - RFECV optimal
    - SHAP top-N for each local minimum in SHAP cutoff curve
    - RFECV local minima (if curve shows multiple good points)
    - Original FULL and SLIM as baselines

    Each candidate logged as a separate MLflow run in price/feature_selection
    with metrics from a reference model (LightGBM defaults).
    """
```

**Key insight from EP:** Different models prefer different feature sets. The notebook showed that both 31 and ~70 features performed similarly on RFECV — local minima exist. We log multiple candidates and let later experiments compare. Crucially, feature selection uses LightGBM as the reference model, but candidate sets should span a range of sizes (SHAP-curated ~30-50, FULL ~100, MAX ~350) so that linear models can pick larger sets where regularization does the selection work.

---

## 5c.3 Price Model Tuning — Grid Search

**`energy_forecasting/modeling/tuning.py`**

Two-stage grid for tree models, exhaustive grid for linear models. EP used ~3 configs per model type and got production-quality results — this is confirmation not exploration. Optuna `GridSampler` provides storage and resumption.

**Key insight: tree models don't need scaler/transform search.** Tree splits are invariant to monotonic feature transforms. Scaler and target transform are `"none"` for trees. `weight_half_life` is the only preprocessing parameter that affects trees, and it's pinned in a cheap first stage.

### Tree Models (LightGBM, XGBoost, CatBoost)

```python
def tune_tree_model(
    dataset_path: Path,
    model_type: str,
) -> dict:
    """Two-stage grid search.

    Stage 1 — pin weight_half_life:
      PRICE_TREE_WEIGHT_PROBE[model_type] (1 config) × WEIGHT_HALF_LIVES (4)
      = 4 trials per model type, 12 total. Pick winning weight.

    Stage 2 — grid over hyperparams with weight fixed:
      PRICE_TREE_GRID[model_type] (8 configs for LightGBM/XGBoost, 5 for CatBoost)
      × 1 weight = 8/8/5 trials.

    Total tree trials: 12 + 21 = 33 per feature set.

    Scaler='none', target_transform='none' (trees are invariant).
    early_stopping_rounds=50 — n_estimators is a ceiling, not a target.

    Optional robustness check: run top 2-3 stage-2 winners under
    runner-up weight to verify the weight choice isn't fragile.
    """
```

### Linear Models (Ridge, Lasso, ElasticNet)

```python
def tune_linear_model(
    dataset_path: Path,
    model_type: str,
) -> dict:
    """Exhaustive grid over full preprocessing × alpha (× l1_ratio for ElasticNet).

    Ridge: 24 preprocessing × 15 alpha = 360 trials.
    Lasso: 24 preprocessing × 12 alpha = 288 trials.
    ElasticNet: 24 preprocessing × 10 alpha × 5 l1_ratio = 1200 trials.
    All at ~1-3s per trial — runs in minutes.
    """
```

### Compute Budget & Performance

**Two-tier CV strategy:**
- **During grid search:** `SEARCH_CV_FOLDS = 3` for fast evaluation.
- **During ensemble validation:** Winning models re-evaluated with `VALIDATION_CV_FOLDS = 5`.

**Early stopping (tree models):**
- `early_stopping_rounds=50` with validation fold as eval set. Most trials stop well below the `n_estimators` ceiling.
- Actual `n_estimators` used is logged as a param.

**Parallelism:**
- **Within each trial:** `n_jobs=-1` for sklearn CV parallelism.
- **Across feature sets × model types:** naturally parallelisable — separate processes, separate Optuna SQLite DBs.

**Optuna storage:**
- SQLite backend: `optuna.storages.RDBStorage(f"sqlite:///data/optuna/{study_name}.db")`.
- Studies survive interruption and resume with `load_study()`.

**Expected times (station, 16 cores). Estimates — first feature-selection run should validate and update this table:**

| Step | Per-trial time | Trials | Total |
|---|---|---|---|
| Feature computation (MAX) | — | 1 | 3-5 min |
| Tree stage 1: weight probe | ~15-45s | 12 | 3-9 min |
| LightGBM stage 2: hyperparam grid | ~15-30s | 8 | 2-4 min |
| XGBoost stage 2: hyperparam grid | ~15-30s | 8 | 2-4 min |
| CatBoost stage 2: hyperparam grid | ~30-45s | 5 | 3-4 min |
| Ridge exhaustive | ~0.5-1.5s | 360 | ~4 min |
| Lasso exhaustive | ~0.5-1.5s | 288 | ~3 min |
| ElasticNet exhaustive | ~0.5-1.5s | 1200 | ~10 min |
| Ensemble validation (24 × 5 folds) | ~15-45s | 120 fits | 0.5-1.5 hr |
| **Total per feature set** | | **~1881** | **~0.75-2 hr** |
| **Total (2 feature sets)** | | | **~1.5-4 hr** |

With parallelism across model types: ~0.5-1.5 hr per feature set.

**Why grid over TPE:** EP's production params define the known-good region. Grid search gives deterministic, complete coverage — every trial is informative by design. The difference between good and great hyperparams is small relative to CV noise, so TPE's adaptive sampling has little signal to exploit. EP used 3 configs per model type and got production-quality results.

**Gen/load models still use TPE** — no strong priors from EP, and the weather FE config space is genuinely exploratory.

### Resuming After Interruption

All heavy steps are designed to be resumable:

- **Optuna studies:** SQLite-backed, resume with `load_study()`. Completed grid points are preserved.
- **MLflow runs:** Append-only. Failed runs visible, successful runs immutable.
- **Feature datasets:** Parquet files on disk. If interrupted, `find_dataset()` returns None and it's recomputed.
- **Ensemble selection:** Reads from MLflow. Re-run `compare_ensemble_methods()` if interrupted.

**Practical recovery:** Re-run the tuning step. Optuna skips completed grid points. MLflow has the finished runs. No manual cleanup needed.

---

## 5c.4 Price Model Training — Precise Workflow

The complete pipeline from raw data to production ensemble:

```
Step 1: Prepare datasets
  └─ For each candidate feature list (SLIM, FULL, MAX, plus SHAP-curated sets):
     prepare_dataset() → Parquet path

Step 2: Feature selection experiment (price/feature_selection)
  └─ run_feature_selection() on MAX dataset (LightGBM reference)
  └─ Produces 4-6 candidate feature sets, each with metrics
  └─ Log all candidates as MLflow runs with full SHAP/RFECV curves
  └─ Ensure range of sizes: small SHAP-curated, medium FULL, large MAX
     (linear models may prefer larger sets with regularization doing selection)

Step 3: Grid search tuning (price/model_training)
  └─ Stage 3a: Pin weight_half_life for tree models
     └─ 1 probe config per tree type × 4 weights = 12 trials (~10 min)
     └─ Pick winning weight (shared across all tree types, or per-type if they disagree)
  └─ Stage 3b: For each of top 2 feature sets:
     └─ Tree models (LightGBM, XGBoost, CatBoost):
        └─ tune_tree_model() — hyperparam grid with weight fixed
        └─ Scaler/target_transform = none (trees are invariant)
        └─ 21 trials total (8 + 8 + 5)
     └─ Linear models (Ridge, Lasso, ElasticNet):
        └─ tune_linear_model() — exhaustive grid over full preprocessing × alpha/l1_ratio
        └─ ~1848 trials total (seconds each)
  └─ Total: 12 + 2 feature sets × ~1869 trials ≈ ~3750 trials
  └─ Uses SEARCH_CV_FOLDS (3) for speed; winners validated with 5 folds later
  └─ Each model type may prefer a different feature set — that's expected

Step 4: Ensemble candidate selection & validation (price/production)
  └─ select_candidates() — 6 per category from price/model_training:
     2 incumbents, 1 best-MAE, 1 best-RMSE, 2 random (logged)
     (first run: best-MAE, best-RMSE, 4 random)
  └─ validate_candidates() — clone & retrain each with VALIDATION_CV_FOLDS (5)
     (24 candidates × 5 folds = 120 fits)
  └─ select_final_models() — pick best from validated candidates
  └─ train_and_blend() — clone & retrain final models on full train set,
     compute predictions on holdout
  └─ compare_ensemble_methods() — all 9 methods on holdout predictions
  └─ select_best_ensemble()
  └─ Calibrate ensemble prediction intervals (post-hoc conformal)
  └─ Save ensemble_config.json

Step 5: Retrain from committed params (for CI)
  └─ load_best_params() → clone & retrain all models from scratch
  └─ Recompute ensemble weights on fresh holdout
  └─ Degradation check vs previous ensemble
```

Each step is an MLflow experiment. Results are persistent and reviewable. Every feature selection decision and tuning trial is logged with full context.

### `ensemble_config.json` Schema

Committed to repo. Contains everything needed to retrain from scratch without MLflow access.

```json
{
  "created": "2026-04-01T12:00:00Z",
  "ensemble_method": "greedy_forward",
  "conformal_quantile": 14.2,
  "blend_mae": 9.93,
  "blend_rmse": 15.12,
  "blend_r2": 0.78,
  "models": [
    {
      "name": "lgbm_0",
      "model_class": "LGBMRegressor",
      "category": "lgbm",
      "source_run_id": "abc123",
      "dataset_path": "data/processed/datasets/price_shap_top40.parquet",
      "feature_version": "shap_top40",
      "feature_list": ["price_h24", "prog_residual", "..."],
      "hyperparams": {"n_estimators": 1000, "learning_rate": 0.012, "...": "..."},
      "preprocessing": {"scaler": "none", "target_transform": "none", "weight_half_life": 730},
      "weight": 0.132,
      "holdout_mae": 10.38,
      "holdout_rmse": 15.75,
      "selection_reason": "incumbent"
    }
  ]
}
```

Key fields per model:
- `feature_list` + `hyperparams` + `preprocessing` = enough to rebuild the model from raw data
- `source_run_id` + `dataset_path` = provenance (run ID for MLflow traceability, path for data)
- `selection_reason` = "incumbent" | "best_mae" | "best_rmse" | "random" (tracks how each model entered the ensemble)
- `weight` = ensemble weight assigned by the selected method

**Inference with multiple feature lists:** Models in the ensemble may use different feature sets (e.g. model A uses SHAP-top-40, model B uses FULL-130). At inference (stage 6), the pipeline must compute the **union** of all feature lists, then each model selects its subset. `engineer_features(df, union_list)` is called once; each model indexes into the result. This avoids redundant computation while allowing per-model feature sets.

---

## 5c.5 Ensemble Methods

**`energy_forecasting/modeling/ensemble.py`** — single module containing all nine methods plus auto-selection. The legacy `modeling/blend.py` and `modeling/stacking.py` stub files are deleted as part of this stage; do not split ensemble logic across multiple files.

**Standing methodology:** all 9 methods are evaluated on every retrain (not just the EP-baseline winner `greedy_forward`). The cost is small once base models are trained, and this lets the production ensemble switch methods automatically if a different one wins on future data — important because the EP ranking was established on a different evaluation window and may not hold as the data evolves. `compare_ensemble_methods()` runs the full set; `select_best_ensemble()` picks by holdout MAE; the chosen method is recorded in `ensemble_config.json` and can change between retrains without code changes.

### Weight-Based Methods

All take `preds_matrix` (n_samples × n_models) and `y_true`:

| Method | Description | EP Result (MAE) |
|---|---|---|
| `simple_average` | Equal weights | Baseline |
| `inverse_mae` | `1/(mae+ε)`, 150-day trailing window | 10.29 |
| `slsqp_optimized` | Scipy SLSQP on OOF MAE, `Σw=1, w≥0` | 10.74 |
| `greedy_forward` | Add best-improving model iteratively | **10.22** |
| `hill_climbing` | Greedy + swap/drop perturbations | 10.40 |
| `simulated_annealing` | Stochastic model selection | 10.37 |
| `diversity_regularized` | Greedy with diversity bonus (α=0.05) | 10.33 |

### Stacking Methods

| Method | Meta-Learner | EP Result (MAE) |
|---|---|---|
| `stacking_ridge` | Ridge(positive=True) | 11.73 |
| `stacking_lgbm` | LGBMRegressor(n_est=100, depth=3) | 12.28 |

### Auto-Selection

```python
def compare_ensemble_methods(base_models, X, y, holdout_days, methods=None) -> pd.DataFrame:
    """Run all methods, return comparison sorted by holdout MAE."""

def select_best_ensemble(comparison) -> tuple[str, Any, dict]:
    """Pick winner. Returns (method_name, fitted_ensemble, config)."""
```

---

# Stage 5d: Analysis Notebooks

Notebooks that read from MLflow to provide qualitative understanding of model behaviour. All are parameterised by run IDs — they never train models.

**Style rule:** These should look like notebooks a human data scientist wrote. Let the plots speak — use concise markdown headers between cells, not verbose ASCII banners or long print statements. DataFrames rendered via `.style` or plain display, not custom print formatting. Minimal code comments; the cell structure itself provides the narrative.

## 5d.1 Feature Selection Analysis (price)

**`notebooks/feature_selection_analysis.ipynb`**

Input: run IDs from `price/feature_selection` experiment.

Cells:
1. **Config cell** — run IDs as parameters, MLflow connection
2. **Candidate set summary** — table of all candidate feature sets with size, reference MAE, RMSE
3. **SHAP summary plot** — beeswarm plot of top-30 features by mean |SHAP|
4. **SHAP dependence plots** — top-10 features, coloured by strongest interaction
5. **Feature correlation heatmap** — clustered heatmap of inter-feature correlations for each candidate set, highlighting redundancy
6. **RFECV curve** — n_features vs CV score, marking local minima (the candidate sets)
7. **Candidate set overlap** — Venn/UpSet diagram showing which features are shared across candidates
8. **Commentary cell** — markdown template for noting observations and decisions

## 5d.2 Model Diagnostics (price + gen/load)

**`notebooks/model_diagnostics.ipynb`**

Input: one or more run IDs from `price/model_training` or `generation/*` experiments. Works for both price and gen/load — the run ID determines which.

Cells:
1. **Config cell** — run IDs, MLflow connection, load holdout predictions + PI bounds
2. **Headline metrics** — table of MAE, RMSE, R², skill score vs baselines
3. **Residuals by hour of day** — boxplot of residuals per hour (reveals systematic hourly bias)
4. **Residuals by day of week** — weekend vs weekday error patterns
5. **Residuals by month/season** — seasonal drift detection
6. **Spike analysis** — scatter of predicted vs actual for extreme values (|value| > P90), annotated with dates
7. **Error time series** — rolling 30-day MAE over the holdout period, to spot if errors cluster in specific periods
8. **PI coverage over time** — rolling 30-day coverage rate, with 90% target line (identifies intervals where PI is too narrow/wide)
9. **PI width vs actual volatility** — scatter of PI width against realised |residual|, checking calibration
10. **Optuna parameter importance** — if Optuna study artifact available, plot parameter importance and parallel coordinate plot
11. **Multi-model comparison** — if multiple run IDs provided, overlay residual distributions and metric comparison bar chart

## 5d.3 Ensemble Analysis (price)

**`notebooks/ensemble_analysis.ipynb`**

Input: run IDs of ensemble candidates + final ensemble from `price/production`.

Cells:
1. **Config cell** — candidate run IDs, ensemble run ID, MLflow connection
2. **Method comparison** — bar chart of all 9 ensemble methods' holdout MAE/RMSE (from the comparison step)
3. **Model diversity heatmap** — pairwise prediction correlation matrix of base models (high diversity = less correlated)
4. **Weight distribution** — bar chart of ensemble weights per base model, grouped by model category (linear/lgbm/xgboost/catboost)
5. **Weight stability across CV folds** — line plot of how each model's weight changes across CV folds (unstable weights = fragile ensemble)
6. **Per-model contribution** — for each holdout hour, which base model's prediction was closest to actual? Stacked area chart showing contribution over time
7. **Ensemble vs best single model** — scatter of ensemble prediction vs best individual model prediction, coloured by which is closer to actual
8. **Residual comparison** — overlay residual distributions of ensemble vs top-3 individual models
9. **When does the ensemble fail?** — top-20 worst holdout hours: what happened? (annotate with date, hour, price level, individual model predictions)

## 5d.4 Gen/Load Analysis

**`notebooks/gen_load_analysis.ipynb`**

Input: run IDs from `generation/*` experiments (one per target/region combination).

Cells:
1. **Config cell** — run IDs per target/region, MLflow connection
2. **Cross-region accuracy table** — MAE, RMSE, R² for each region within a target (e.g. wind_onshore across 4 TSOs), highlighting hardest regions
3. **Cross-region error correlation** — heatmap of residual correlations across regions (correlated errors suggest a shared missing signal, e.g. a weather pattern not captured)
4. **Forecast horizon decay** — MAE by forecast day (day 1 through day 7), per target. How fast does accuracy degrade?
5. **Weather FE importance** — SHAP or permutation importance of weather-derived features, showing which physical variables (irradiance, wind speed, temperature) drive each target
6. **Spatial aggregation effect** — if multiple aggregation methods were tested, compare accuracy (helps decide whether to keep capacity-weighted vs simple mean)
7. **Seasonal patterns** — accuracy by month for each target (solar should be worse in shoulder seasons, wind more uniform)

---

## New / Modified Files Summary

### Stage 5a
| File | Est. Lines | Purpose |
|---|---|---|
| `config/modeling.py` | +80 | All constants with comments |
| `config/search_spaces.py` | ~180 | Search spaces, grid points, preprocessing values |
| `modeling/mlflow_utils.py` | ~220 | TrackedRun (validates at enter), auto-tags, helpers |
| `modeling/metrics.py` | ~80 | RMSE, MAE, skill, PI metrics |
| `modeling/cv.py` | ~130 | TimeSeriesSplitter |
| `modeling/training.py` | ~220 | Pipeline builder, train_model, sample weights |
| `modeling/intervals.py` | ~90 | MAPIE wrapper + post-hoc conformal |
| `modeling/datasets.py` | ~120 | prepare/load/update/find Parquet datasets, MLflow provenance |
| `modeling/baselines.py` | ~60 | Naive, weekly, SARIMA |
| `modeling/forecasting.py` | ~100 | Direct (default) + recursive forecast |

### Stage 5b
| File | Est. Lines | Purpose |
|---|---|---|
| `modeling/gen_load.py` | ~350 | Gen/load pipeline, Optuna, stacking ensemble, baselines |

### Stage 5c
| File | Est. Lines | Purpose |
|---|---|---|
| `config/features.py` | +220 | `PRICE_FEATURES_MAX` |
| `config/columns.py` | +15 | 13 new `SHORT_NAMES` entries (forecasts, eeg_regime, neg_price_*, per-tech pct_prog) |
| `config/availability.py` | +3 | New rules for `forecast_*`, `neg_price_*`, `eeg_regime` |
| `features/market.py` | +20 | Extend `compute_generation_pct()` for `pct_prog_solar/wind_on/wind_off`; add `compute_eeg_regime()`, `compute_neg_price_stats()` |
| `features/engine.py` | +30 | Three new `_prepare_working_df()` branches: forecasts, eeg_regime, neg_price |
| `modeling/gen_load_forecasts.py` | ~50 | `load_gen_load_forecasts()` — direct-parquet loader for `data/processed/historical_forecasts/`, with `forecast_residual` derivation and missing-file pre-flight check |
| `modeling/feature_selection.py` | ~250 | Correlation, SHAP, RFECV, multiple candidates |
| `modeling/price.py` | ~150 | Price training pipeline |
| `modeling/tuning.py` | ~220 | Two-phase grid (trees) + exhaustive grid (linear) via Optuna GridSampler |
| `modeling/ensemble.py` | ~400 | All 9 methods + stacking + auto-select |
| `modeling/blend.py` | delete | Stub removed — functionality in `ensemble.py` |
| `modeling/stacking.py` | delete | Stub removed — functionality in `ensemble.py` |
| `cli.py` | +60 | train, tune, select-features, blend commands |
| `Makefile` | +15 | Corresponding targets |

### Stage 5d (deferred — token-intensive)
| File | Est. Lines | Purpose |
|---|---|---|
| `notebooks/feature_selection_analysis.ipynb` | ~200 | SHAP plots, correlation heatmaps, candidate set comparison (price) |
| `notebooks/model_diagnostics.ipynb` | ~250 | Residuals, spikes, PI coverage (price + gen/load) |
| `notebooks/ensemble_analysis.ipynb` | ~200 | Model diversity, weight stability, per-model contribution (price) |
| `notebooks/gen_load_analysis.ipynb` | ~200 | Cross-region comparison, forecast horizon decay, weather FE importance |

These notebooks read from MLflow (parameterised by run IDs) — they don't do training. Pattern: **training in code, analysis in notebooks**.

---

## MLflow Artifact Logging Requirements

Training code must log sufficient artifacts for 5d notebooks to work without re-running models:

| Artifact | Logged by | Used by |
|---|---|---|
| SHAP values (per feature, all samples) | `feature_selection.py` | feature_selection_analysis |
| RFECV curve (n_features → score) | `feature_selection.py` | feature_selection_analysis |
| OOF predictions (all CV folds) | `training.py` | model_diagnostics |
| Holdout predictions + PI bounds | `training.py` | model_diagnostics, ensemble_analysis |
| Hourly residuals on holdout | `training.py` | model_diagnostics |
| Optuna study object | `tuning.py` | model_diagnostics (param importance) |
| Ensemble weight matrix | `ensemble.py` | ensemble_analysis |
| Per-model holdout predictions | `ensemble.py` | ensemble_analysis |

---

## Tests

| Test file | Coverage |
|---|---|
| `tests/test_metrics.py` | All metrics, skill scores, PI metrics |
| `tests/test_cv.py` | Day boundaries, expanding/sliding, holdout excluded |
| `tests/test_training.py` | Pipeline, train loop, transforms, weights, MLflow |
| `tests/test_intervals.py` | MAPIE wrap, fit/predict, post-hoc conformal, XGBoost eps |
| `tests/test_datasets.py` | prepare/load/update/find, Parquet round-trip, log_input |
| `tests/test_forecasting.py` | Direct default, recursive with lags, continuity |
| `tests/test_feature_selection.py` | Correlation, SHAP, RFECV, multiple candidates |
| `tests/test_ensemble.py` | All 9 methods, auto-selection, degradation |
| `tests/test_baselines.py` | Naive, weekly, shape/alignment |
| `tests/test_tuning.py` | Two-phase grid, exhaustive linear grid, param save/load |

---

## Gates

Each sub-stage ends with a gate review. Present the evidence below and wait for user acceptance before proceeding.

### Gate 5a → 5b

**Evidence to present:**
1. Test suite output — all 5a tests pass (`test_metrics`, `test_cv`, `test_training`, `test_intervals`, `test_datasets`, `test_forecasting`, `test_baselines`)
2. Demo: `train_model()` end-to-end on synthetic data showing MLflow run with correct tags, metrics, and artifacts logged
3. Demo: `TrackedRun` rejecting a run with missing required tags
4. Demo: `TimeSeriesSplitter` fold boundaries (print first/last timestamp of each train/test fold — verify day boundaries)
5. Demo: MAPIE `predict_with_intervals()` producing (point, lower, upper) with correct shapes
6. Answers to open questions #1-3 (MAPIE empirical checks — resolved during implementation)

**Acceptance criteria:**
- All tests green
- Training loop logs all artifacts from the artifact table (SHAP values, OOF predictions, holdout predictions + PI, residuals, Optuna study)
- `config/modeling.py` has no magic numbers — every constant has a comment explaining where it's used
- Two-tier CV works: `SEARCH_CV_FOLDS=3` for grid search iterations, `VALIDATION_CV_FOLDS=5` for final evaluation

### Gate 5b → 5c

**Evidence to present:**
1. Test suite output — `test_gen_load` passes
2. Per-target/region accuracy table: compare against EMA reference numbers (include MAE, RMSE, R² for each)
3. MAPIE PI coverage per target — should be ~90% ± 3%
4. Weather FE Parquet files exist in `data/processed/weather_features/` for all targets/regions
5. Answer to open question #4 (gen/load dependency order — resolved during implementation)

**Acceptance criteria:**
- Accuracy within 5% of EMA reference for each target/region (or explanation of why a gap exists)
- PI coverage 87-93% across targets
- Weather FE artifacts saved and loadable for price feature transfer

**Before 5c — pre-flight checklist:**

1. Verify all expected files exist in `data/processed/historical_forecasts/` (16 files total):
   - `wind_onshore_DE_{50HZ, AMPRION, TENNET, TRANSNETBW}.parquet` + `wind_onshore_DE_NATIONAL.parquet` (4 TSO + 1 NAT)
   - `wind_offshore_DE_{50HZ, TENNET}.parquet` + `wind_offshore_DE_NATIONAL.parquet` (2 TSO + 1 NAT)
   - `solar_DE_{50HZ, AMPRION, TENNET, TRANSNETBW}.parquet` + `solar_DE_NATIONAL.parquet` (4 TSO + 1 NAT)
   - `load_DE_{50HZ, AMPRION, CREOS, TENNET, TRANSNETBW}.parquet` + `load_DE_NATIONAL.parquet` (5 TSO + 1 NAT)
   - `gen_load_diff_DE_NATIONAL.parquet` (NAT only)
2. Spot-check schema on one file: columns are `[y_true, y_pred, y_lower, y_upper]`, indexed by timestamp.
3. Note: `y_lower`/`y_upper` are NaN for upstream base models (recursive-lag models bypass MAPIE). The price loader treats those as missing PI, not missing predictions.

### Gate 5c → 5d

**Evidence to present:**
1. Test suite output — `test_feature_selection`, `test_tuning`, `test_ensemble` pass
2. Feature selection summary: candidate sets with sizes and reference-model MAE for each
3. Grid search summary: best params per model type (including which feature set and weight_half_life each tree model chose, and full preprocessing for linear models)
4. Ensemble comparison table: all 9 methods with holdout MAE, RMSE
5. Best ensemble MAE vs EP production reference (~9.9 EUR/MWh)
6. Ensemble PI coverage
7. `ensemble_config.json` contents
8. Answer to open question #5 (EEG regime features — resolved during implementation)

**Acceptance criteria:**
- Feature selection produces ≥3 candidate sets spanning small/medium/large sizes
- Price ensemble MAE ≤ 11 EUR/MWh (within ~10% of EP production; exact parity not required as data period differs)
- Ensemble PI coverage 87-93%
- `ensemble_config.json` is complete and can drive a retrain from scratch

### Gate 5d (final)

**Evidence to present:**
1. Each notebook renders without errors when pointed at real run IDs from 5b/5c
2. Walk through key plots from each notebook

**Acceptance criteria:**
- Notebooks are parameterised (run ID input cell at top) and work on any valid run
- Feature selection notebook shows SHAP summary, dependence plots, correlation heatmap, candidate comparison
- Model diagnostics notebook shows residuals by hour/weekday/season, spike analysis, PI coverage over time — works for both price and gen/load run IDs
- Ensemble notebook shows diversity heatmap, weight stability, per-model contribution
- Gen/load notebook shows cross-region comparison, forecast horizon decay, weather FE importance
- Notebooks look like a human wrote them — no verbose ASCII banners, no long print statements, plots and DataFrames do the talking

---

## Open Questions

1. **MAPIE 1.3 + XGBoost eps:** ~~Verify empirically whether `CrossConformalRegressor` needs the eps fix.~~ **RESOLVED (5a):** Not needed. MAPIE 1.3 handles XGBoost float32 natively. Tested: coverage=96%, no NaN, all bounds valid.

2. **MAPIE + sample_weight pass-through:** ~~Verify `fit_conformalize(fit_params={"model__sample_weight": w})` works through the pipeline.~~ **RESOLVED (5a):** Pipeline routing doesn't work — MAPIE passes full weight vector to internal CV folds without subsetting. Solution: two-path approach. Without weights → Pipeline directly to MAPIE. With weights → pre-scale X externally, pass bare estimator with `fit_params={"sample_weight": w}`.

3. **Single-row predict_interval:** ~~If recursive forecasting is activated, verify MAPIE 1.3 supports single-row predictions.~~ **RESOLVED (5a):** Works. `predict_interval` on shape (1, N) returns (1,) point and (1, 2, 1) intervals.

4. **Gen/load dependency order:** ~~Load model may use wind/solar forecasts as exogenous features (EMA pattern). If so: wind → solar → load training order.~~ **RESOLVED (5b):** Yes, dependency order required. Load models use wind/solar OOF predictions as exog features (EMA pattern). `gen_load_diff` (national) uses wind/solar/load predictions. Training order enforced: `wind_onshore, wind_offshore, solar` (parallel) → `load` → `gen_load_diff`. Upstream OOF + holdout predictions loaded via `_load_upstream_predictions()` from MLflow artifacts. All TSO regions of each upstream target are included as exog features (10 features for load, 15 for gen_load_diff).

5. **EEG regime features:** ~~Need new SHORT_NAMES and availability rules for `eeg_regime`, `neg_price_frac_*`, `neg_price_depth_*`.~~ **RESOLVED (5c plan review, 2026-04-10):** Three derived columns with different treatment. `eeg_regime` is deterministic (offset=0, bare name, computed from `EEG_REGIME_DATES` dates). `neg_price_frac_30d`, `neg_price_frac_90d`, `neg_price_depth_30d` are rolling stats on the target price — SHORT_NAMES map to `_derived_neg_price_*` columns, availability rule `neg_price_*` → offset=-1. Feature strings in `PRICE_FEATURES_MAX` use the `_d1` suffix (`neg_price_frac_30d_d1`, etc.) so the engine picks up yesterday's value of the pre-computed rolling column. `_prepare_working_df()` gets two new branches (one for `eeg_regime`, one for `neg_price_*`). Full specification in §5c.0.

6. **Interaction features with two suffixed sides:** ~~`price_d2_d1_std__x__price_d7_d1_std` is an interaction between two aggregation features. The parser should handle this (each side parsed independently), but verify during PRICE_FEATURES_MAX implementation.~~ **RESOLVED (5c plan review, 2026-04-10):** Parser handles this correctly. `parse_feature()` splits on `__x__` and recurses on each side independently. Each side parses to a `FeatureSpec` with an `Aggregation` (left: start=-2, end=-1, stat='std'; right: start=-7, end=-1, stat='std'). Both validate against the `price_*` availability rule (offset=-1). No code changes needed — just add the feature strings to `PRICE_FEATURES_MAX` and add a test in `test_parser.py` to lock in this behavior.

7. **Optuna storage directory:** ~~`data/optuna/` doesn't exist in the current project structure.~~ **RESOLVED (5c plan refinement, 2026-05-05):** `data/optuna/` is not yet present (5b stored its Optuna study elsewhere). `modeling/tuning.py` will `Path("data/optuna").mkdir(parents=True, exist_ok=True)` on first use — no separate setup step needed.

8. **Weather features in price models:** ~~Two approaches: two-step vs direct; decide after gen/load models exist.~~ **RESOLVED (5c plan review, 2026-04-10; refined 2026-05-05):** Use the two-step approach. Gen/load forecasts enter price models as `forecast_*` derived columns loaded directly from on-disk parquet at `data/processed/historical_forecasts/` (§5c.0 "Gen/load forecast integration" — note: refined from the original MLflow-artifact-load architecture once 5b shipped and pinned the on-disk schema). Direct weather features are deferred: if price ensemble MAE gaps vs EP production after 5c, revisit by adding top-signal weather features (national irradiance, wind speed) to `PRICE_FEATURES_MAX`. This keeps the feature set compact and side-steps the per-TSO aggregation problem.

9. **Missing pct_prog_* per-technology derived columns:** `pct_prog_solar`, `pct_prog_wind_on`, `pct_prog_wind_off` appear in `PRICE_FEATURES_MAX` but `compute_generation_pct()` only emits `_derived_pct_prog_other` and `_derived_pct_prog_wind_pv`. The raw SMARD prognosis columns (`prognostizierte_erzeugung_onshore/offshore/photovoltaik`) exist and are already mapped to `prog_gen_wind_on/off/solar` in `config/columns.py` — the gap is only in the derived-pct computation and SHORT_NAMES. See §5c.0 "Extending `compute_generation_pct()`". Three-line code change plus three new SHORT_NAMES entries.
