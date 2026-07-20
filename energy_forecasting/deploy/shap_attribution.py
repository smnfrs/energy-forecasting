"""Per-instance SHAP attribution for the production price ensemble.

Computed inside the daily inference pipeline (price_inference.py), not in
narrative.py — it's cheap, deterministic, and network-free, so it belongs in
the always-runs path, riding along with the existing feature_audit persistence.

Each production price model is a MAPIE ``CrossConformalRegressor``. Its point
prediction (``model.predict(X)``, called with the default
``aggregate_predictions="mean"``) is a *weighted* mean of the predictions of
its 5 cross-validation fold estimators — see
``mapie.estimator.regressor.EnsembleRegressor._aggregate_with_mask``. The
weight of fold k is the fraction of training rows whose designated
out-of-fold estimator is fold k (``est.k_``), which is close to but not
exactly 1/5 per fold. To reproduce ``y_pred`` exactly, SHAP values and base
values are computed per fold estimator and combined with those same weights,
before being combined again across the ensemble's own per-model weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

# Fixed attribution categories, checked in order — first keyword match wins.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("gas", ("ttf", "gen_gas", "gen_pct_gas")),
    ("carbon", ("carbon",)),
    ("oil", ("brent",)),
    (
        "neighbour_prices",
        tuple(
            f"price_{cc}"
            for cc in ("at", "be", "ch", "cz", "dk1", "dk2", "fr", "nl", "no2", "pl", "se4")
        ),
    ),
    ("wind", ("wind",)),
    ("solar", ("solar", "gen_pv", "forecast_gen_solar", "pct_forecast_solar")),
    ("residual_gen", ("residual", "supply_demand_gap", "gen_other", "forecast_gen_other")),
    (
        "conventional_gen",
        (
            "gen_coal", "gen_lignite", "gen_nuclear", "gen_hydro", "gen_biomass",
            "gen_pumped", "pct_renewable", "gen_pct_",
        ),
    ),
    ("cross_border", ("net_export", "total_imports", "total_exports")),
    ("load", ("load",)),
    ("price_momentum", ("price_", "neg_price_")),
    (
        "calendar",
        ("day_of_week", "day_index", "dow_", "hour_", "month_", "fourier_", "is_holiday", "is_weekend"),
    ),
]


def categorize_feature(col: str) -> str:
    """Bucket a feature column name into a fixed driver category by keyword match."""
    for category, keywords in _CATEGORY_RULES:
        if any(kw in col for kw in keywords):
            return category
    return "other"


def _fold_weights(k_matrix: np.ndarray) -> np.ndarray:
    """Weight of each cross-validation fold estimator in the ensemble's mean prediction.

    ``k_matrix`` has shape (n_train_rows, n_folds); each row has a single 1 marking
    which fold is that row's out-of-fold estimator. The weight of fold k is the
    fraction of training rows assigned to it.
    """
    counts = np.nan_to_num(k_matrix, nan=0.0).sum(axis=0)
    return counts / counts.sum()


def _unwrap_pipeline_model(fold_estimator):
    """Return the underlying estimator when a fold estimator is a Pipeline."""
    if hasattr(fold_estimator, "named_steps"):
        return fold_estimator.named_steps["model"]
    return fold_estimator


def _linear_terms(fold_estimator, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Return transformed design matrix, coefficients, and intercept for a linear fold.

    Exported Ridge folds can be either bare estimators or sklearn Pipelines.
    For Pipelines, MAPIE prediction applies all preprocessing steps before the
    final linear model, so attribution must use that same transformed matrix to
    preserve the prediction reconstruction invariant.
    """
    linear_model = _unwrap_pipeline_model(fold_estimator)
    if hasattr(fold_estimator, "steps") and len(fold_estimator.steps) > 1:
        x_values = np.asarray(fold_estimator[:-1].transform(X))
    else:
        x_values = X.to_numpy()
    return x_values, np.asarray(linear_model.coef_), float(linear_model.intercept_)


def _model_shap(
    mapie_model,
    X: pd.DataFrame,
    is_linear: bool,
) -> tuple[np.ndarray, float]:
    """SHAP values + base value for one production model, aggregated across its 5 CV folds.

    Returns (shap_matrix, base_value) where shap_matrix has shape (n_rows, n_features)
    aligned to X.columns, and base_value is a scalar such that
    base_value + shap_matrix.sum(axis=1) == mapie_model.predict(X) (up to float tolerance).
    """
    ensemble = mapie_model._mapie_regressor.estimator_
    fold_estimators = ensemble.estimators_
    weights = _fold_weights(ensemble.k_)

    shap_sum = np.zeros((len(X), X.shape[1]))
    base_sum = 0.0
    for fold_estimator, w in zip(fold_estimators, weights):
        if is_linear:
            x_values, coefs, fold_base = _linear_terms(fold_estimator, X)
            fold_shap = x_values * coefs[np.newaxis, :]
        else:
            tree_model = _unwrap_pipeline_model(fold_estimator)
            explainer = shap.TreeExplainer(tree_model)
            fold_shap = np.asarray(explainer.shap_values(X))
            fold_base = float(np.ravel(explainer.expected_value)[0])
        shap_sum += w * fold_shap
        base_sum += w * fold_base

    return shap_sum, base_sum


def compute_price_shap(
    model_entries: dict[str, dict],
    loaded_models: dict[str, object],
    feature_matrices: dict[str, pd.DataFrame],
    weights: dict[str, float],
    used_weights: list[float],
    used_names: list[str],
) -> dict:
    """Compute ensemble-level SHAP attribution for the D+1 price forecast.

    Parameters mirror what run_price_inference already has in scope after its
    per-model prediction loop: model_entries (name -> ensemble_config entry),
    loaded_models (name -> the loaded MAPIE model object), feature_matrices
    (model name -> its own X_d1 DataFrame, in the same — possibly scaled —
    space used for that model's prediction; keyed by model name rather than
    feature_version since two production models can share a feature_version
    but use different scaler instances), weights (raw ensemble weights dict),
    used_weights/used_names (the normalized weights actually applied, in the
    same order).

    Returns a JSON-serializable dict: per-hour timestamp, base value, and
    signed per-category contribution (EUR/MWh), plus the raw per-feature
    breakdown for the invariant test / debugging.
    """
    w_norm = np.array(used_weights)
    w_norm = w_norm / w_norm.sum()

    # Union of feature columns across all production models' feature-sets,
    # since fs_rfecv_optimum / fs_shap_top90 / fs_shap_top247 are not identical.
    all_columns: list[str] = []
    seen = set()
    for name in used_names:
        for c in feature_matrices[name].columns:
            if c not in seen:
                seen.add(c)
                all_columns.append(c)

    n_rows = len(next(iter(feature_matrices.values())))
    ensemble_shap = pd.DataFrame(0.0, index=range(n_rows), columns=all_columns)
    ensemble_base = 0.0

    for name, w in zip(used_names, w_norm):
        X = feature_matrices[name]
        is_linear = "Ridge" in name or "Lasso" in name

        model_shap, model_base = _model_shap(loaded_models[name], X, is_linear=is_linear)
        model_shap_df = pd.DataFrame(model_shap, index=range(n_rows), columns=X.columns)

        ensemble_shap[X.columns] += w * model_shap_df
        ensemble_base += w * model_base

    # Bucket into fixed categories, summing signed contributions per hour.
    category_by_col = {c: categorize_feature(c) for c in all_columns}
    categories = sorted(set(category_by_col.values()))
    category_matrix = pd.DataFrame(0.0, index=range(n_rows), columns=categories)
    for col, cat in category_by_col.items():
        category_matrix[cat] += ensemble_shap[col]

    return {
        "base_value": float(ensemble_base),
        "categories": categories,
        "category_contributions": {
            cat: [round(float(v), 4) for v in category_matrix[cat]] for cat in categories
        },
        "feature_columns": all_columns,
        "feature_shap": ensemble_shap.round(6).to_numpy().tolist(),
    }
