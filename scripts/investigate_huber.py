"""Stabilize HuberRegressor. The default tiny alpha diverged on early
expanding-CV folds (cv_mae ~1040). Sweep alpha x scaler and check whether a
well-regularized Huber is both stable and a useful diverse ensemble member.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor

from energy_forecasting.config.modeling import HOLDOUT_DAYS, SEARCH_CV_FOLDS
from energy_forecasting.config.search_spaces import PRICE_TREE_WEIGHT_PROBE
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import load_dataset
from energy_forecasting.modeling.metrics import calculate_metrics
from energy_forecasting.modeling.tuning import _make_model
from energy_forecasting.modeling.training import build_pipeline

DATASET = "data/processed/datasets/price_fs_rfecv_optimum.parquet"


def main():
    X, y = load_dataset(DATASET)
    pool_idx, hold_idx = carve_holdout(X.index, HOLDOUT_DAYS)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    yh = y_hold.to_numpy()
    cv = TimeSeriesSplitter(n_splits=SEARCH_CV_FOLDS, mode="expanding")

    # XGB reference errors for diversity measure.
    from energy_forecasting.modeling.training import build_pipeline as bp
    xp = bp(_make_model("XGBRegressor", PRICE_TREE_WEIGHT_PROBE["XGBRegressor"]),
            scaler="none", target_transform="none")
    xp.fit(X_pool, y_pool)
    xgb_err = xp.predict(X_hold) - yh

    print(f"{'scaler':<9} {'alpha':>7} {'cv_mae':>9} {'hold_mae':>9} {'corr_xgb':>9}")
    print("-" * 48)
    for scaler in ("standard", "robust"):
        for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
            fold_maes = []
            ok = True
            for tr, te in cv.split(X_pool.index):
                pipe = build_pipeline(
                    HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=3000),
                    scaler=scaler, target_transform="none")
                try:
                    pipe.fit(X_pool.iloc[tr], y_pool.iloc[tr])
                except Exception:
                    ok = False
                    break
                fold_maes.append(calculate_metrics(y_pool.iloc[te], pipe.predict(X_pool.iloc[te]))["mae"])
            if not ok:
                print(f"{scaler:<9} {alpha:>7} {'FAILED':>9}")
                continue
            pipe = build_pipeline(
                HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=3000),
                scaler=scaler, target_transform="none")
            pipe.fit(X_pool, y_pool)
            pred = pipe.predict(X_hold)
            mae = calculate_metrics(yh, pred)["mae"]
            corr = float(np.corrcoef(pred - yh, xgb_err)[0, 1])
            print(f"{scaler:<9} {alpha:>7} {np.mean(fold_maes):>9.3f} {mae:>9.3f} {corr:>9.3f}")


if __name__ == "__main__":
    main()
