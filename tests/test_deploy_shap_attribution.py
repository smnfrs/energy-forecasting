"""Tests for deploy/shap_attribution.py."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from energy_forecasting.deploy.shap_attribution import categorize_feature, compute_price_shap
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def test_categorize_feature_known_columns():
    assert categorize_feature("ttf_ewma_24_d2") == "gas"
    assert categorize_feature("carbon_d7_d2_avg") == "carbon"
    assert categorize_feature("brent_d2") == "oil"
    assert categorize_feature("price_fr_h24") == "neighbour_prices"
    assert categorize_feature("price_h24") == "price_momentum"
    assert categorize_feature("gen_wind_on_d2") == "wind"
    assert categorize_feature("gen_solar_d2") == "solar"
    assert categorize_feature("residual_load_d2") == "residual_gen"
    assert categorize_feature("gen_pct_hydro_d2") == "conventional_gen"
    assert categorize_feature("net_export_fr_d2") == "cross_border"
    assert categorize_feature("load_d2") == "load"
    assert categorize_feature("day_of_week") == "calendar"
    assert categorize_feature("totally_unknown_xyz") == "other"


def _fake_mapie_model(fold_estimators, k_matrix):
    """Minimal stand-in for MAPIE's CrossConformalRegressor, exposing only the
    attributes _model_shap actually reads: ._mapie_regressor.estimator_.{estimators_,k_}."""
    ensemble = SimpleNamespace(estimators_=fold_estimators, k_=k_matrix)
    mapie_regressor = SimpleNamespace(estimator_=ensemble)
    return SimpleNamespace(_mapie_regressor=mapie_regressor)


def _fold_weighted_pred(folds, k_matrix, X):
    counts = np.nan_to_num(k_matrix, nan=0.0).sum(axis=0)
    w = counts / counts.sum()
    preds = np.column_stack([f.predict(X) for f in folds])
    return (preds * w).sum(axis=1)


def test_compute_price_shap_reconstructs_ensemble_prediction():
    """SHAP invariant: base_value + sum(per-feature shap) must equal the ensemble's
    own weighted-mean-of-folds prediction, for both a linear and a tree model."""
    rng = np.random.default_rng(0)
    n_train, n_test, n_folds = 200, 24, 5
    cols = ["ttf_ewma_24_d2", "price_h24", "gen_wind_on_d2"]
    X_train = pd.DataFrame(rng.normal(size=(n_train, len(cols))), columns=cols)
    y_train = X_train.sum(axis=1) + rng.normal(scale=0.01, size=n_train)
    X_test = pd.DataFrame(rng.normal(size=(n_test, len(cols))), columns=cols)

    # Each training row's designated out-of-fold estimator (mirrors mapie's k_ matrix).
    fold_of_row = rng.integers(0, n_folds, size=n_train)
    k_matrix = np.full((n_train, n_folds), np.nan)
    for i, f in enumerate(fold_of_row):
        k_matrix[i, f] = 1.0

    ridge_folds = [
        Pipeline([("scaler", RobustScaler()), ("model", Ridge(alpha=0.01))]).fit(
            X_train[fold_of_row != f], y_train[fold_of_row != f]
        )
        for f in range(n_folds)
    ]
    lgbm_folds = [
        LGBMRegressor(n_estimators=20, min_child_samples=2, verbose=-1).fit(
            X_train[fold_of_row != f], y_train[fold_of_row != f]
        )
        for f in range(n_folds)
    ]

    model_entries = {
        "Ridge__fv": {"feature_version": "fv"},
        "LGBMRegressor__fv": {"feature_version": "fv"},
    }
    loaded_models = {
        "Ridge__fv": _fake_mapie_model(ridge_folds, k_matrix),
        "LGBMRegressor__fv": _fake_mapie_model(lgbm_folds, k_matrix),
    }
    feature_matrices = {"Ridge__fv": X_test, "LGBMRegressor__fv": X_test}
    weights = {"Ridge__fv": 0.4, "LGBMRegressor__fv": 0.6}
    used_weights = [0.4, 0.6]
    used_names = ["Ridge__fv", "LGBMRegressor__fv"]

    result = compute_price_shap(
        model_entries, loaded_models, feature_matrices, weights, used_weights, used_names
    )

    y_ridge = _fold_weighted_pred(ridge_folds, k_matrix, X_test)
    y_lgbm = _fold_weighted_pred(lgbm_folds, k_matrix, X_test)
    y_blend = 0.4 * y_ridge + 0.6 * y_lgbm

    # atol reflects LGBM's internal float32 prediction precision, not the SHAP math
    # itself — the exact production run (5-fold x2 real models) showed ~1.7e-5.
    recon = np.asarray(result["base_value"]) + np.array(result["feature_shap"]).sum(axis=1)
    np.testing.assert_allclose(recon, y_blend, atol=1e-4)

    assert set(result["categories"]) == {"gas", "price_momentum", "wind"}
    for cat_values in result["category_contributions"].values():
        assert len(cat_values) == n_test
