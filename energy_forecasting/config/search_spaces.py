"""Search spaces and grid definitions for model tuning.

Based on EP's production parameters (from blend_hyperparams.json).
Ranges narrowed around what actually worked — high regularization, moderate learning rates.
"""

import itertools

import numpy as np

# ── Optuna suggest functions (for gen/load TPE search) ─────────────

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
        "objective": "mae",  # fixed — EP found MAE loss best
        "metric": "mae",  # early stopping monitors MAE
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
        "eval_metric": "mae",  # early stopping monitors MAE
    }


def suggest_catboost(trial) -> dict:
    """CatBoost. EP production: lr=0.01-0.02, depth=8-9, l2_leaf_reg=5-7."""
    return {
        "iterations": trial.suggest_int("iterations", 800, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 8.0),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "rsm": trial.suggest_float("rsm", 0.45, 0.75),  # colsample equivalent
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
        "loss_function": "MAE",  # fixed — EP found MAE loss best
        "eval_metric": "MAE",  # early stopping monitors MAE
        "verbose": 0,  # fixed — suppress CatBoost logging
    }


def suggest_ridge(trial) -> dict:
    """Ridge. EP production: alpha=0.1."""
    return {"alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True)}


def suggest_lasso(trial) -> dict:
    """Lasso. EP production: alpha=0.1."""
    return {"alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True)}


def suggest_dataset_params(trial, model_type: str | None = None) -> dict:
    """Gen/load dataset-level suggestions — searched jointly with model params.

    `model_type` opts the trial out of param values that are known-incompatible
    with a given model. ElasticNet pins `log_target=False` because RobustScaler
    on log-transformed targets produces values that overflow ElasticNet's
    coordinate descent (matches EMA's hardcoded `log_target: False` for
    ElasticNet at `update_forecasts.py:93–104`).
    """
    if model_type == "ElasticNet":
        log_target = False
    else:
        log_target = trial.suggest_categorical("log_target", [True, False])
    return {
        "log_target": log_target,
        "lags_target": trial.suggest_categorical("lags_target", [None, 1, 6, 12]),
        "scaler": trial.suggest_categorical("scaler", ["standard", "robust", "minmax"]),
    }


# ── Price model grid points ───────────────────────────────────────
# Two-stage grid for trees:
#   Stage 1: Pin weight_half_life using 1 representative config per model type.
#   Stage 2: Grid over hyperparams with winning weight fixed.

# Stage 1: weight selection — 1 full config per type × 4 weights = 12 trials total.
# These are EP-production-quality configs so weight comparison is fair.
#
# LGBM capacity note (fixed 2026-06-06): num_leaves must scale with max_depth.
# A fixed num_leaves=31 caps a leaf-wise tree at ~depth-5 capacity regardless
# of max_depth, starving LGBM relative to XGB (depth-wise, ~256 leaves at
# max_depth=8) and CatBoost (symmetric, 256 leaves at depth=8). We let
# max_depth be the limiter (num_leaves set to the full 2^depth-1) and enable
# bagging_freq so `subsample` is not a silent no-op. Objective stays MAE for
# EP comparability — see docs/stage5c_status_2026-06-06.md (testing L2 across
# all tree families is a roadmap item).
PRICE_TREE_WEIGHT_PROBE = {
    "LGBMRegressor": {
        "learning_rate": 0.012,
        "n_estimators": 1000,
        "max_depth": 8,
        "num_leaves": 255,  # = 2^8 - 1; was 31 (capacity bug, fixed 2026-06-06)
        "min_child_samples": 50,
        "subsample": 0.7,
        "subsample_freq": 1,  # without this LGBM ignores `subsample`
        "colsample_bytree": 0.7,
        "reg_alpha": 5.0,
        "reg_lambda": 5.0,
        "min_split_gain": 0.1,
        "objective": "mae",
        "metric": "mae",
    },
    "XGBRegressor": {
        "learning_rate": 0.02,
        "n_estimators": 1000,
        "max_depth": 8,
        "min_child_weight": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 5.0,
        "reg_lambda": 5.0,
        "gamma": 0.3,
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
    },
    "CatBoostRegressor": {
        "learning_rate": 0.015,
        "iterations": 1000,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "subsample": 0.7,
        "rsm": 0.7,
        "min_child_samples": 50,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "verbose": 0,
    },
}

# Stage 2: hyperparam grid with weight fixed.
# LGBM configs set num_leaves = 2^max_depth - 1 so max_depth is the real
# limiter (see capacity note above). XGB/CatBoost grids only override depth
# because their growth is depth-bounded by construction.
PRICE_TREE_GRID = {
    "LGBMRegressor": [
        {"learning_rate": 0.008, "max_depth": 8, "num_leaves": 255, "reg_alpha": 5.0, "n_estimators": 1000},
        {"learning_rate": 0.015, "max_depth": 6, "num_leaves": 63, "reg_alpha": 7.0, "n_estimators": 800},
        {"learning_rate": 0.012, "max_depth": 8, "num_leaves": 255, "reg_alpha": 10.0, "n_estimators": 1000},
        {"learning_rate": 0.02, "max_depth": 10, "num_leaves": 1023, "reg_alpha": 3.0, "n_estimators": 1200},
        {"learning_rate": 0.008, "max_depth": 7, "num_leaves": 127, "reg_alpha": 5.0, "n_estimators": 800},
        {"learning_rate": 0.01, "max_depth": 9, "num_leaves": 511, "reg_alpha": 8.0, "n_estimators": 1100},
        {"learning_rate": 0.025, "max_depth": 6, "num_leaves": 63, "reg_alpha": 5.0, "n_estimators": 900},
        {"learning_rate": 0.008, "max_depth": 8, "num_leaves": 255, "reg_alpha": 12.0, "n_estimators": 1000},
    ],
    "XGBRegressor": [
        {"learning_rate": 0.023, "max_depth": 8, "reg_alpha": 5.0, "n_estimators": 1000},
        {"learning_rate": 0.015, "max_depth": 6, "reg_alpha": 8.0, "n_estimators": 800},
        {"learning_rate": 0.01, "max_depth": 8, "reg_alpha": 10.0, "n_estimators": 1000},
        {"learning_rate": 0.025, "max_depth": 10, "reg_alpha": 3.0, "n_estimators": 1200},
        {"learning_rate": 0.015, "max_depth": 7, "reg_alpha": 6.0, "n_estimators": 900},
        {"learning_rate": 0.02, "max_depth": 9, "reg_alpha": 8.0, "n_estimators": 1100},
        {"learning_rate": 0.01, "max_depth": 6, "reg_alpha": 12.0, "n_estimators": 800},
        {"learning_rate": 0.03, "max_depth": 8, "reg_alpha": 4.0, "n_estimators": 1000},
    ],
    "CatBoostRegressor": [
        {"learning_rate": 0.015, "depth": 8, "l2_leaf_reg": 5.0, "iterations": 1000},
        {"learning_rate": 0.01, "depth": 9, "l2_leaf_reg": 7.0, "iterations": 800},
        {"learning_rate": 0.02, "depth": 7, "l2_leaf_reg": 5.0, "iterations": 1200},
        {"learning_rate": 0.012, "depth": 8, "l2_leaf_reg": 8.0, "iterations": 1000},
        {"learning_rate": 0.018, "depth": 6, "l2_leaf_reg": 4.0, "iterations": 900},
    ],  # CatBoost is 2-3× slower — 5 configs is enough
}
# Tree trials per dataset: stage1 4 weights each + stage2 grid
# (LGBM 8 + XGB 8 + CatBoost 5) + 3×4 weight probes.

# ── Linear model grids ───────────────────────────────────────────
LINEAR_ALPHA_GRID = {
    "Ridge": np.logspace(-3, 1, 15).tolist(),  # 0.001 to 10
    "Lasso": np.logspace(-3, 0, 9).tolist(),   # 0.001 to 1
}

# ── Preprocessing ─────────────────────────────────────────────────
TARGET_TRANSFORMS = ["none"]  # log_shift / yeo_johnson dropped — empirically no gain
FEATURE_SCALERS = ["standard", "robust"]  # "none" omitted — linear models require scaling
WEIGHT_HALF_LIVES = [None, 365, 730, 1095]

# Tree models: no scaler, no target transform (invariant to monotonic transforms).
# Only weight_half_life is searched.
TREE_PREPROCESSING = {"scaler": "none", "target_transform": "none"}

# Linear models: scaler × weight_half_life grid. Target transform is pinned
# to "none" (dropped as a search axis — empirically no gain, and log/yeo
# blew up Ridge/Lasso), so it is no longer part of the tuple.
LINEAR_PREPROCESSING_GRID = list(itertools.product(
    FEATURE_SCALERS, WEIGHT_HALF_LIVES,
))  # 2 × 4 = 8 combos
