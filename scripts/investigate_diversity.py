"""Quick read on candidate models before the full pipeline run.

Tests, on price_fs_rfecv_optimum (n=40), 3-fold expanding CV + 90d holdout:
  * fixed LGBM (mae kept for EP comparability; num_leaves scaled, bagging on)
  * HuberRegressor (robust linear, scaled)
  * quantile GBM (LGBM objective=quantile alpha=0.5)

Diversity value = decorrelation of a model's holdout *errors* from XGB's.
A model only helps a tree-heavy ensemble if it is both decent AND adds error
diversity. Reports holdout-error correlation with the XGB reference and the
MAE of a naive 50/50 XGB+candidate blend.
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


def fit_predict(model, scaler, X_pool, y_pool, X_hold, cv):
    """Return (cv_mae, holdout_pred) for a model spec."""
    fold_maes = []
    for tr, te in cv.split(X_pool.index):
        pipe = build_pipeline(_clone_spec(model), scaler=scaler, target_transform="none")
        pipe.fit(X_pool.iloc[tr], y_pool.iloc[tr])
        fold_maes.append(calculate_metrics(y_pool.iloc[te], pipe.predict(X_pool.iloc[te]))["mae"])
    pipe = build_pipeline(_clone_spec(model), scaler=scaler, target_transform="none")
    pipe.fit(X_pool, y_pool)
    return float(np.mean(fold_maes)), pipe.predict(X_hold)


def _clone_spec(spec):
    mt, params = spec
    if mt == "Huber":
        return HuberRegressor(**params)
    return _make_model(mt, params)


def main():
    X, y = load_dataset(DATASET)
    pool_idx, hold_idx = carve_holdout(X.index, HOLDOUT_DAYS)
    X_pool, y_pool = X.iloc[pool_idx], y.iloc[pool_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]
    yh = y_hold.to_numpy()
    cv = TimeSeriesSplitter(n_splits=SEARCH_CV_FOLDS, mode="expanding")

    lp = PRICE_TREE_WEIGHT_PROBE["LGBMRegressor"]
    lgbm_fixed = {**lp, "num_leaves": 255, "max_depth": -1, "bagging_freq": 1}
    qgbm = {**lp, "num_leaves": 255, "max_depth": -1, "bagging_freq": 1,
            "objective": "quantile", "alpha": 0.5, "metric": "quantile"}

    # XGB reference (no scaler).
    xgb_cv, xgb_pred = fit_predict(
        ("XGBRegressor", PRICE_TREE_WEIGHT_PROBE["XGBRegressor"]), "none",
        X_pool, y_pool, X_hold, cv)
    xgb_err = xgb_pred - yh
    xgb_mae = calculate_metrics(yh, xgb_pred)["mae"]
    print(f"XGB reference: cv_mae={xgb_cv:.3f}  holdout_mae={xgb_mae:.3f}\n")

    candidates = [
        ("LGBM fixed (mae,leaves=255,bag1)", ("LGBMRegressor", lgbm_fixed), "none"),
        ("LGBM quantile (a=0.5)",            ("LGBMRegressor", qgbm),       "none"),
        ("HuberRegressor (robust scaler)",   ("Huber", {"alpha": 0.001, "epsilon": 1.35, "max_iter": 2000}), "robust"),
    ]

    print(f"{'model':<36} {'cv_mae':>8} {'hold_mae':>9} {'err_corr_xgb':>13} {'50/50_xgb':>10}")
    print("-" * 80)
    for label, spec, scaler in candidates:
        cv_mae, pred = fit_predict(spec, scaler, X_pool, y_pool, X_hold, cv)
        mae = calculate_metrics(yh, pred)["mae"]
        err = pred - yh
        corr = float(np.corrcoef(err, xgb_err)[0, 1])
        blend_mae = calculate_metrics(yh, 0.5 * (pred + xgb_pred))["mae"]
        print(f"{label:<36} {cv_mae:>8.3f} {mae:>9.3f} {corr:>13.3f} {blend_mae:>10.3f}")
    print("\nLower err_corr_xgb = more diversity. 50/50_xgb < 11.49 means the "
          "candidate improves a pure-XGB blend even at a crude equal weight.")


if __name__ == "__main__":
    main()
