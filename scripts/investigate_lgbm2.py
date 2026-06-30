"""LGBM investigation part 2: objective, bagging_freq, learning rate.

Two more canonical LightGBM gotchas beyond num_leaves:
  1. objective="mae" -> L1 has zero hessian; LightGBM split-finding and leaf
     values degrade vs L2. XGB's reg:absoluteerror is better engineered for it.
  2. subsample=0.7 is a NO-OP in LightGBM unless bagging_freq>0. So the probe
     thinks it row-subsamples but doesn't; XGB *does* subsample.
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
    fold_maes = []
    t0 = time.time()
    for tr, te in cv.split(X_pool.index):
        m = _make_model(model_type, params)
        pipe = build_pipeline(m, scaler="none", target_transform="none")
        pipe.fit(X_pool.iloc[tr], y_pool.iloc[tr])
        fold_maes.append(calculate_metrics(y_pool.iloc[te], pipe.predict(X_pool.iloc[te]))["mae"])
    m = _make_model(model_type, params)
    pipe = build_pipeline(m, scaler="none", target_transform="none")
    pipe.fit(X_pool, y_pool)
    hold_mae = calculate_metrics(y_hold, pipe.predict(X_hold))["mae"]
    return float(np.mean(fold_maes)), hold_mae, time.time() - t0


def main():
    X, y = load_dataset(DATASET)
    pool_idx, hold_idx = carve_holdout(X.index, HOLDOUT_DAYS)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    cv = TimeSeriesSplitter(n_splits=SEARCH_CV_FOLDS, mode="expanding")

    lp = PRICE_TREE_WEIGHT_PROBE["LGBMRegressor"]
    # Best capacity config from part 1.
    cap = {**lp, "num_leaves": 255, "max_depth": -1}

    configs = [
        ("LGBM probe (mae, num_leaves=31)", "LGBMRegressor", {**lp}),
        ("LGBM cap (mae, leaves=255,d=-1)", "LGBMRegressor", {**cap}),
        ("LGBM cap + objective=regression(L2)", "LGBMRegressor",
         {**cap, "objective": "regression", "metric": "l2"}),
        ("LGBM cap + objective=huber", "LGBMRegressor",
         {**cap, "objective": "huber", "metric": "l1"}),
        ("LGBM cap + bagging_freq=1 (subsample active)", "LGBMRegressor",
         {**cap, "bagging_freq": 1}),
        ("LGBM cap + L2 + bagging_freq=1", "LGBMRegressor",
         {**cap, "objective": "regression", "metric": "l2", "bagging_freq": 1}),
        ("LGBM cap + L2 + bag1 + lr=0.02", "LGBMRegressor",
         {**cap, "objective": "regression", "metric": "l2", "bagging_freq": 1,
          "learning_rate": 0.02}),
        ("LGBM cap + L2 + bag1 + lr=0.03 + n=1500", "LGBMRegressor",
         {**cap, "objective": "regression", "metric": "l2", "bagging_freq": 1,
          "learning_rate": 0.03, "n_estimators": 1500}),
    ]

    print(f"n_features={X.shape[1]}  pool={len(X_pool)}  cv_folds={SEARCH_CV_FOLDS}\n")
    print(f"{'config':<48} {'cv_mae':>8} {'hold_mae':>9} {'secs':>7}")
    print("-" * 75)
    for label, mt, params in configs:
        cv_mae, hold_mae, dt = evaluate(mt, params, X_pool, y_pool, X_hold, y_hold, cv)
        print(f"{label:<48} {cv_mae:>8.3f} {hold_mae:>9.3f} {dt:>7.1f}")
    print("\nref: XGB probe hold_mae=11.49, ensemble winner hold_mae=11.24")


if __name__ == "__main__":
    main()
