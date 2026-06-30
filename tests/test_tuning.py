"""Tests for modeling/tuning.py — grid search helpers.

The integration paths (``tune_tree_model``/``tune_linear_model``) require
a prepared dataset on disk + MLflow + Optuna SQLite + GBT fits, so they
are exercised end-to-end by ``train price`` runs rather than here. These
tests cover the building blocks: model factory, CV evaluator, search-space
construction, and the MLflow tag routing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.modeling.cv import TimeSeriesSplitter
from energy_forecasting.modeling.tuning import (
    PRICE_LINEAR_TYPES,
    PRICE_TREE_TYPES,
    _evaluate_config,
    _linear_search_space,
    _make_model,
)


# ── Model factory ────────────────────────────────────────────────


@pytest.mark.parametrize("model_type", PRICE_TREE_TYPES + PRICE_LINEAR_TYPES)
def test_make_model_returns_correct_class(model_type):
    if model_type == "CatBoostRegressor":
        params = {"iterations": 5, "verbose": 0}
    elif model_type == "XGBRegressor":
        params = {"n_estimators": 5}
    elif model_type == "LGBMRegressor":
        params = {"n_estimators": 5}
    else:
        params = {"alpha": 0.1}
    model = _make_model(model_type, params)
    assert type(model).__name__ == model_type


def test_make_model_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        _make_model("NotAModel", {})


def test_make_model_threads_through_jobs_flag_for_trees():
    # Trees should default to n_jobs=-1 for parallelism.
    m = _make_model("LGBMRegressor", {"n_estimators": 5})
    assert m.n_jobs == -1
    m = _make_model("XGBRegressor", {"n_estimators": 5})
    assert m.n_jobs == -1


# ── Linear search space ──────────────────────────────────────────


def test_linear_search_space_ridge_two_dims():
    space = _linear_search_space("Ridge")
    assert set(space.keys()) == {"preproc_idx", "alpha_idx"}
    # Search-space sizes match the registered grid
    from energy_forecasting.config.search_spaces import (
        LINEAR_ALPHA_GRID,
        LINEAR_PREPROCESSING_GRID,
    )
    assert len(space["preproc_idx"]) == len(LINEAR_PREPROCESSING_GRID)
    assert len(space["alpha_idx"]) == len(LINEAR_ALPHA_GRID["Ridge"])


def test_linear_search_space_lasso_two_dims():
    space = _linear_search_space("Lasso")
    assert set(space.keys()) == {"preproc_idx", "alpha_idx"}


def test_linear_search_space_unsupported_model_raises():
    # _linear_search_space dispatches off LINEAR_ALPHA_GRID, so anything
    # outside that is a KeyError at lookup time.
    with pytest.raises(KeyError):
        _linear_search_space("LGBMRegressor")


# ── CV evaluator ─────────────────────────────────────────────────


@pytest.fixture
def tiny_xy():
    idx = pd.date_range("2024-01-01", periods=24 * 90, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=len(idx)),
            "f1": rng.normal(size=len(idx)),
            "f2": rng.normal(size=len(idx)),
        },
        index=idx,
    )
    y = pd.Series(2 * X["f0"] - X["f1"] + rng.normal(0, 0.5, len(idx)), index=idx)
    return X, y


def test_evaluate_config_returns_cv_prefixed_metrics(tiny_xy):
    X, y = tiny_xy
    cv = TimeSeriesSplitter(n_splits=3, mode="expanding")
    metrics = _evaluate_config(
        X, y, cv,
        model_type="Ridge",
        model_params={"alpha": 0.1},
        scaler="standard",
        target_transform="none",
        weight_half_life=None,
    )
    assert "cv_mae" in metrics
    assert "cv_rmse" in metrics
    assert metrics["cv_mae"] > 0


def test_evaluate_config_applies_weight_half_life(tiny_xy):
    X, y = tiny_xy
    cv = TimeSeriesSplitter(n_splits=3, mode="expanding")
    # Without weights
    m_no_w = _evaluate_config(
        X, y, cv,
        model_type="Ridge",
        model_params={"alpha": 0.1},
        scaler="standard",
        target_transform="none",
        weight_half_life=None,
    )
    # With weights (short half-life — recent data dominates)
    m_w = _evaluate_config(
        X, y, cv,
        model_type="Ridge",
        model_params={"alpha": 0.1},
        scaler="standard",
        target_transform="none",
        weight_half_life=30.0,
    )
    # Both succeed; metrics differ (weights change the fit)
    assert m_no_w["cv_mae"] != m_w["cv_mae"]


def test_evaluate_config_tree_with_none_scaler(tiny_xy):
    """Tree models pin scaler='none', target_transform='none' per the plan."""
    X, y = tiny_xy
    cv = TimeSeriesSplitter(n_splits=3, mode="expanding")
    metrics = _evaluate_config(
        X, y, cv,
        model_type="LGBMRegressor",
        model_params={"n_estimators": 50},
        scaler="none",
        target_transform="none",
        weight_half_life=None,
    )
    assert metrics["cv_mae"] > 0
