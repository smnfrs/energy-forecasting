"""Investigate LGBM underperformance vs XGB/CatBoost in the price pipeline.

Hypothesis: LGBM's num_leaves is pinned at 31 (from the probe config) across
the whole grid, so it is capacity-starved relative to XGB (max_depth=8 grows
depth-wise to ~256 leaves) and CatBoost (depth=8 -> 256 leaves). num_leaves=31
corresponds to a fully-grown tree of depth ~5.

Runs the *actual* code paths (load_dataset, carve_holdout, build_pipeline,
calculate_metrics) on the winning feature set, comparing LGBM variants.
"""

from __future__ import annotations

import time

import numpy as np

from energy_forecasting.config.modeling import HOLDOUT_DAYS, SEARCH_CV_FOLDS
from energy_forecasting.config.search_spaces import PRICE_TREE_WEIGHT_PROBE
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import load_dataset
from energy_forecasting.modeling.metrics import calculate_metrics
from energy_forecasting.modeling.tuning import _make_model
from energy_forecasting.modeling.training import build_pipeline

DATASET = "data/processed/datasets/price_fs_rfecv_optimum.parquet"


def evaluate(model_type, params, X_pool, y_pool, X_hold, y_hold, cv):
    """CV mean MAE + refit holdout MAE for one config. Trees: no scaler."""
    fold_maes = []
    t0 = time.time()
    for tr, te in cv.split(X_pool.index):
        m = _make_model(model_type, params)
        pipe = build_pipeline(m, scaler="none", target_transform="none")
        pipe.fit(X_pool.iloc[tr], y_pool.iloc[tr])
        pred = pipe.predict(X_pool.iloc[te])
        fold_maes.append(calculate_metrics(y_pool.iloc[te], pred)["mae"])
    # Refit on full pool, eval on holdout.
    m = _make_model(model_type, params)
    pipe = build_pipeline(m, scaler="none", target_transform="none")
    pipe.fit(X_pool, y_pool)
    hold_mae = calculate_metrics(y_hold, pipe.predict(X_hold))["mae"]
    dt = time.time() - t0
    return float(np.mean(fold_maes)), hold_mae, dt


def main():
    X, y = load_dataset(DATASET)
    pool_idx, hold_idx = carve_holdout(X.index, HOLDOUT_DAYS)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    cv = TimeSeriesSplitter(n_splits=SEARCH_CV_FOLDS, mode="expanding")
    print(f"dataset={DATASET}  n_features={X.shape[1]}  "
          f"pool={len(X_pool)}  holdout={len(X_hold)}  cv_folds={SEARCH_CV_FOLDS}\n")

    lgbm_probe = PRICE_TREE_WEIGHT_PROBE["LGBMRegressor"]
    xgb_probe = PRICE_TREE_WEIGHT_PROBE["XGBRegressor"]
    cat_probe = PRICE_TREE_WEIGHT_PROBE["CatBoostRegressor"]

    print(f"LGBM probe num_leaves={lgbm_probe['num_leaves']}, "
          f"max_depth={lgbm_probe['max_depth']}  "
          f"(2^max_depth={2**lgbm_probe['max_depth']} leaves theoretically available)\n")

    configs = [
        # (label, model_type, params)
        ("LGBM probe (num_leaves=31, current)", "LGBMRegressor", {**lgbm_probe}),
        ("LGBM num_leaves=63",  "LGBMRegressor", {**lgbm_probe, "num_leaves": 63}),
        ("LGBM num_leaves=127", "LGBMRegressor", {**lgbm_probe, "num_leaves": 127}),
        ("LGBM num_leaves=255 (=depth8)", "LGBMRegressor", {**lgbm_probe, "num_leaves": 255}),
        ("LGBM num_leaves=255 depth=-1 (unbounded)", "LGBMRegressor",
         {**lgbm_probe, "num_leaves": 255, "max_depth": -1}),
        ("XGB probe (max_depth=8)", "XGBRegressor", {**xgb_probe}),
        ("CatBoost probe (depth=8)", "CatBoostRegressor", {**cat_probe}),
    ]

    print(f"{'config':<45} {'cv_mae':>8} {'hold_mae':>9} {'secs':>7}")
    print("-" * 72)
    for label, mt, params in configs:
        cv_mae, hold_mae, dt = evaluate(mt, params, X_pool, y_pool, X_hold, y_hold, cv)
        print(f"{label:<45} {cv_mae:>8.3f} {hold_mae:>9.3f} {dt:>7.1f}")


if __name__ == "__main__":
    main()
