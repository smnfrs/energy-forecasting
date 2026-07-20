"""Modelling constants: experiment names, model categories, ensemble defaults.

All defaults live here with comments explaining where they're used.
No magic numbers anywhere else in the modeling code.

Ported from EP's src/config/modeling.py with extensions for gen/load targets.
"""

# ── Feature contract ───────────────────────────────────────
# Forecast-feature epoch used for MLflow comparability after replacing leaky
# SMARD prog_* price features with source-neutral forecast_* columns.
FEATURE_CONTRACT = "forecast_v1"

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

# Minimum own-gen/load forecast artifact coverage required over the exact
# price-training holdout split before ensemble blending or conformal
# calibration may report metrics. The release target is stricter by
# convention (prefer >= 0.99), but 0.95 is the default hard guard.
PRICE_HOLDOUT_MIN_OWN_FORECAST_FRACTION = 0.95

# Monthly all-five-artifact coverage guard used in daily deploy before price
# inference starts. Months below this own-source fraction indicate a
# historical_forecasts gap that would push price features onto SMARD fallback.
FORECAST_ARTIFACT_MIN_MONTHLY_OWN_FRACTION = 0.95

# ── Sample weighting ──────────────────────────────────────────────
# Exponential decay half-life in days. At half_life, weight = 0.5.
# 730 days = 2 years. Used by compute_sample_weights() in training.py.
DEFAULT_WEIGHT_HALF_LIFE = 730.0

# ── Ensemble candidate selection ──────────────────────────────────
# After final validation runs are available, keep the best OOF-MAE and
# best OOF-RMSE model per available category. The two metrics can select the
# same model, so the realised count per category is one or two.
ENSEMBLE_FINAL_PER_CATEGORY = 2
ENSEMBLE_MIN_MEMBER_WEIGHT = 0.02
ENSEMBLE_MAX_ALIGNMENT_DROP_FRACTION = 0.05

# ── Degradation detection ─────────────────────────────────────────
# If (new_mae - old_mae) / old_mae exceeds this, flag needs_reselection.
# Used by deploy/retrain.py.
ENSEMBLE_DEGRADATION_THRESHOLD = 0.20

# ── Diagnostic ensemble method registry ───────────────────────────
# Consumed by scripts/ensemble_method_comparison.py only. Production uses the
# EP-faithful inverse-MAE recent-holdout blend and does not select from this
# registry.
ENSEMBLE_METHODS = [
    "simple_average",
    "inverse_mae",
    "inverse_rmse",
    "top_k_trimmed",
    "slsqp_optimized",
    "slsqp_floor_2pct",
    "greedy_forward",
    "greedy_forward_floor_2pct",
    "hill_climbing",
    "hill_climbing_floor_2pct",
    "simulated_annealing",
    "simulated_annealing_floor_2pct",
    "diversity_regularized",
    "stacking_ridge",
    "stacking_lgbm",
]

# ── Gen/load targets ──────────────────────────────────────────────
# Training order matters: load uses wind/solar forecasts as features,
# gen_load_diff uses wind/solar/load. See GEN_LOAD_TRAINING_ORDER.
GEN_LOAD_TARGETS = {
    "wind_onshore": {
        "regions": ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNETBW"],
        "exog_targets": [],
    },
    "wind_offshore": {
        "regions": ["DE_50HZ", "DE_TENNET"],
        "exog_targets": [],
    },
    "solar": {
        "regions": ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNETBW"],
        "exog_targets": [],
    },
    "load": {
        "regions": [
            "DE_50HZ",
            "DE_AMPRION",
            "DE_TENNET",
            "DE_TRANSNETBW",
            "DE_CREOS",
        ],
        # Actual values used during training; forecast outputs used at inference.
        "exog_targets": ["wind_onshore", "wind_offshore", "solar"],
    },
    "gen_load_diff": {
        # National-level only — sum(generation) - sum(load) for DE/LU.
        "regions": ["DE_NATIONAL"],
        "exog_targets": ["wind_onshore", "wind_offshore", "solar", "load"],
    },
}
GEN_LOAD_HORIZON_HOURS = 168  # 7 days

# Training order: targets in the same group are independent and can run in
# parallel. Each group must complete before the next starts.
GEN_LOAD_TRAINING_ORDER: list[list[str]] = [
    ["wind_onshore", "wind_offshore", "solar"],  # independent
    ["load"],  # depends on wind/solar
    ["gen_load_diff"],  # depends on all above
]

# Map region codes (used in GEN_LOAD_TARGETS) to TSO directory names
# (used in data/processed/tso/ and data/raw/weather/).
REGION_TO_TSO: dict[str, str] = {
    "DE_50HZ": "50Hertz",
    "DE_AMPRION": "Amprion",
    "DE_TENNET": "TenneT",
    "DE_TRANSNETBW": "TransnetBW",
    "DE_CREOS": "Creos",
    "DE_NATIONAL": "national",  # gen_load_diff: aggregated from all TSOs
}

# Map gen/load target names to weather asset types for locations_for_tso()
# and weather data directory names.
TARGET_WEATHER_TYPE: dict[str, str] = {
    "wind_onshore": "onshore",
    "wind_offshore": "offshore",
    "solar": "solar",
    "load": "cities",
    "gen_load_diff": "cities",  # same weather FE as load (EMA pattern)
}

# Default Optuna trial count for gen/load models.
# TPE needs ~50-100 trials to converge with this search space dimensionality.
GEN_LOAD_OPTUNA_TRIALS = 70

# ── Gen/load training window ─────────────────────────────────────
# Total dataset length retained before CV. Sized so that a sliding
# `GEN_LOAD_HISTORICAL_FOLDS`-week test span fits within the cap with a
# meaningful per-fold training window remaining.
#
# Layout: n_days = sliding_train_days + n_splits * step_days
# With 233 weekly folds (1631 days) and 51,000 hours (2125 days), each
# fold trains on the preceding ~480 days of actual-weather data.
#
# History was 16,800 (= 700 days, 1.92 years) when GEN_LOAD_HISTORICAL_FOLDS
# was 40, matching EMA's `n_horizons * horizon`. Bumped 2026-05-05 to enable
# OOF coverage back to early 2022 (start of `hist_forecast` weather), then
# 2026-07-14 to close the Apr-Jun 2026 coverage hole without shrinking the
# sliding train window.
GEN_LOAD_MAX_TRAIN_HOURS = 51_000  # 2,125 days ≈ 5.82 years

# Test-fold length for gen/load CV (matches EMA's `horizon=168`).
# Combined with sliding mode and step_days=test_days, each CV fold covers
# exactly one week of test data. See EMA `utils.py:compute_timeseries_split_cutoffs`.
GEN_LOAD_CV_TEST_DAYS = 7

# Holdout reserved for final evaluation of gen/load models. EMA has no
# explicit holdout — its evaluation is the last N CV test weeks. We keep a
# small 1-week holdout as a sanity check beyond EMA, aligned so that the
# CV still covers the most recent weeks below the holdout.
GEN_LOAD_HOLDOUT_DAYS = 7

# Number of CV folds in the *final* (validation) training pass. Each fold
# produces 1 week of OOF predictions; the concatenation is saved as the
# historical-forecasts artifact consumed by Stage 5c.
#
# 233 weekly folds = 1,631 days ≈ 4.47 years of OOF coverage. This preserves
# the full 2022-01-15+ own-forecast history while extending coverage through
# the 2026-04 -> 2026-06 hole found during the forecast_v1 price retrain. The
# earliest test fold stays inside `hist_forecast` weather availability
# (2022-01-01) so all OOF rows are evaluated against forecast weather, the
# EMA `mode="backtest"` regime. Was 40 folds (~9.5 months) before
# 2026-05-05 and 218 folds before the 2026-07-14 coverage remediation.
#
# This collapses EMA's two-pipeline approach (stacking-base OOF +
# `generate_historical_forecasts.py` multi-year backtests) into a single
# OOF stream produced as a byproduct of final-model training. The Optuna
# search keeps the smaller SEARCH_CV_FOLDS for speed; only the final pass
# needs full coverage.
GEN_LOAD_HISTORICAL_FOLDS = 233

# ── Price model categories ────────────────────────────────────────
PEAK_HOURS = list(range(8, 20))
ENSEMBLE_CATEGORY_MATCHERS = {
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
    "gen_gen_load_diff": "generation/gen_load_diff",
}

# ── EEG regime dates ──────────────────────────────────────────────
# §51 EEG negative price clawback thresholds.
# Used to create regime indicator features.
EEG_4H_RULE_DATE = "2023-01-01"  # 6h → 4h threshold
EEG_2H_RULE_DATE = "2024-01-01"  # 4h → 2h (interim)
EEG_SOLARSPITZENGESETZ_DATE = "2025-02-25"  # Any negative 15-min block

# Mapping consumed by compute_eeg_regime() — order matters (ascending dates).
# Regime integers:
#   0 = pre-2023 (no negative-price clawback rule active)
#   1 = 4h threshold (2023-01-01 → 2024-01-01)
#   2 = 2h threshold (2024-01-01 → 2025-02-25)
#   3 = Solarspitzengesetz (any negative 15-min block, 2025-02-25 onwards)
EEG_REGIME_DATES: list[tuple[str, int]] = [
    (EEG_4H_RULE_DATE, 1),
    (EEG_2H_RULE_DATE, 2),
    (EEG_SOLARSPITZENGESETZ_DATE, 3),
]
