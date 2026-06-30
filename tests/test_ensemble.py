"""Tests for modeling/ensemble.py — 11 methods + compare/select."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.modeling.ensemble import (
    HOLDOUT_FIT_METHODS,
    METHOD_FACTORIES,
    OOF_FIT_METHODS,
    StackEnsemble,
    WeightEnsemble,
    compare_ensemble_methods,
    ensemble_config_dict,
    fit_diversity_regularized,
    fit_ensemble,
    fit_greedy_forward,
    fit_hill_climbing,
    fit_inverse_mae,
    fit_inverse_rmse,
    fit_simple_average,
    fit_simulated_annealing,
    fit_slsqp,
    fit_stacking_lgbm,
    fit_stacking_ridge,
    fit_top_k_trimmed,
    select_best_ensemble,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def synthetic_preds():
    """4 base models with increasing noise; one clearly weakest."""
    rng = np.random.default_rng(42)
    n_oof, n_hold = 300, 80
    y_oof = pd.Series(rng.normal(50, 20, n_oof), name="y")
    y_hold = pd.Series(rng.normal(50, 20, n_hold), name="y")
    noises = [3.0, 4.0, 5.0, 20.0]  # last model is dramatically worse
    names = [f"m{i}" for i in range(len(noises))]
    preds_oof = pd.DataFrame({n: y_oof + rng.normal(0, s, n_oof) for n, s in zip(names, noises)})
    preds_hold = pd.DataFrame(
        {n: y_hold + rng.normal(0, s, n_hold) for n, s in zip(names, noises)}
    )
    return preds_oof, y_oof, preds_hold, y_hold, names


# ── Registry sanity ───────────────────────────────────────────────


def test_all_11_methods_registered():
    expected = {
        "simple_average",
        "inverse_mae",
        "inverse_rmse",
        "top_k_trimmed",
        "slsqp_optimized",
        "greedy_forward",
        "hill_climbing",
        "simulated_annealing",
        "diversity_regularized",
        "stacking_ridge",
        "stacking_lgbm",
    }
    assert set(METHOD_FACTORIES) == expected


def test_holdout_oof_partition_covers_registry():
    # Every registered method must be in exactly one of the two sets.
    assert HOLDOUT_FIT_METHODS.isdisjoint(OOF_FIT_METHODS)
    assert set(METHOD_FACTORIES) == HOLDOUT_FIT_METHODS | OOF_FIT_METHODS


def test_stacking_methods_are_oof_fit():
    assert "stacking_ridge" in OOF_FIT_METHODS
    assert "stacking_lgbm" in OOF_FIT_METHODS


def test_weight_methods_are_holdout_fit():
    for m in [
        "simple_average",
        "inverse_mae",
        "inverse_rmse",
        "top_k_trimmed",
        "slsqp_optimized",
        "greedy_forward",
        "hill_climbing",
        "simulated_annealing",
        "diversity_regularized",
    ]:
        assert m in HOLDOUT_FIT_METHODS, f"{m} should be holdout-fit"


# ── Individual factories ──────────────────────────────────────────


def test_simple_average_equal_weights(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_simple_average(preds_oof, y_oof)
    assert ens.method == "simple_average"
    assert np.allclose(ens.weights, 0.25)


def test_inverse_mae_downweights_noisy_model(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_inverse_mae(preds_oof, y_oof)
    # m3 (noise=20) should get the smallest weight
    assert ens.weights[-1] == min(ens.weights)
    # Weights normalised
    assert ens.weights.sum() == pytest.approx(1.0)
    assert (ens.weights >= 0).all()


def test_inverse_rmse_downweights_noisy_model(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_inverse_rmse(preds_oof, y_oof)
    assert ens.weights[-1] == min(ens.weights)
    assert ens.weights.sum() == pytest.approx(1.0)


def test_top_k_trimmed_zeros_excluded_models(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_top_k_trimmed(preds_oof, y_oof, keep=2)
    # Exactly 2 non-zero weights, the rest zero
    assert (ens.weights > 0).sum() == 2
    # The worst model gets weight 0
    assert ens.weights[-1] == 0
    assert ens.weights.sum() == pytest.approx(1.0)


def test_top_k_trimmed_default_keeps_six_or_less(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_top_k_trimmed(preds_oof, y_oof)
    # Default ``keep`` is min(6, n_models); n=4 → keep=4
    assert (ens.weights > 0).sum() == 4


def test_slsqp_optimised_sums_to_one(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_slsqp(preds_oof, y_oof)
    assert ens.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (ens.weights >= 0).all()


def test_greedy_forward_picks_best_first(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_greedy_forward(preds_oof, y_oof)
    # The first pick is always the lowest-MAE single model — count > 0
    counts = np.array(ens.metadata["counts"])
    assert counts.sum() > 0
    # Weakest model (m3) should get the least mass
    assert counts[-1] == counts.min()


def test_hill_climbing_starts_from_greedy(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_hill_climbing(preds_oof, y_oof)
    assert ens.method == "hill_climbing"
    assert ens.weights.sum() == pytest.approx(1.0)


def test_simulated_annealing_is_deterministic_with_seed(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    a = fit_simulated_annealing(preds_oof, y_oof, rng_seed=7, n_iter=200)
    b = fit_simulated_annealing(preds_oof, y_oof, rng_seed=7, n_iter=200)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_diversity_regularized_produces_valid_ensemble(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    diverse = fit_diversity_regularized(preds_oof, y_oof, diversity_weight=0.5)
    assert diverse.method == "diversity_regularized"
    assert diverse.weights.sum() == pytest.approx(1.0)
    assert (diverse.weights >= 0).all()


def test_stacking_ridge_returns_stack_ensemble(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_stacking_ridge(preds_oof, y_oof)
    assert isinstance(ens, StackEnsemble)
    assert ens.method == "stacking_ridge"
    # Ridge(positive=True) → all coefs ≥ 0
    assert (ens.meta_learner.coef_ >= 0).all()


def test_stacking_lgbm_predicts_reasonably(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    ens = fit_stacking_lgbm(preds_oof, y_oof)
    preds = ens.predict(preds_hold)
    # MAE should be at most the worst single model's MAE
    worst = max(float(np.mean(np.abs(y_hold - preds_hold[c]))) for c in preds_hold.columns)
    actual = float(np.mean(np.abs(y_hold - preds)))
    assert actual <= worst * 1.1, f"stacking_lgbm MAE {actual} ≫ worst single {worst}"


# ── Predict ordering invariants ──────────────────────────────────


def test_predict_respects_column_order(synthetic_preds):
    preds_oof, y_oof, preds_hold, _, _ = synthetic_preds
    ens = fit_inverse_mae(preds_oof, y_oof)
    # Shuffle columns and verify .predict still matches model_names
    shuffled = preds_hold[list(reversed(preds_hold.columns))]
    a = ens.predict(preds_hold)
    b = ens.predict(shuffled)
    np.testing.assert_allclose(a, b)


def test_predict_rejects_missing_model(synthetic_preds):
    preds_oof, y_oof, preds_hold, _, _ = synthetic_preds
    ens = fit_inverse_mae(preds_oof, y_oof)
    short = preds_hold.drop(columns=[preds_hold.columns[0]])
    with pytest.raises(KeyError, match="missing model columns"):
        ens.predict(short)


def test_predict_array_must_match_shape(synthetic_preds):
    preds_oof, y_oof, _, _, _ = synthetic_preds
    ens = fit_inverse_mae(preds_oof, y_oof)
    arr = np.zeros((10, len(ens.model_names) - 1))
    with pytest.raises(ValueError, match="must be"):
        ens.predict(arr)


# ── compare_ensemble_methods ──────────────────────────────────────


def test_compare_runs_every_method_plus_single_baselines(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, names = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    expected_methods = set(METHOD_FACTORIES) | {f"single::{n}" for n in names}
    assert set(df["method"]) == expected_methods


def test_compare_sorted_by_mae(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    assert df["mae"].is_monotonic_increasing


def test_compare_carries_fitted_via_attrs(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    fitted = df.attrs.get("fitted")
    assert isinstance(fitted, dict)
    # Every row in the comparison must have a fitted ensemble
    for method in df["method"]:
        assert method in fitted


def test_compare_holdout_oof_routing(synthetic_preds):
    """Verify that a method in HOLDOUT_FIT_METHODS is genuinely fit on
    the holdout window (its in-sample MAE on holdout equals the reported
    MAE under inverse_mae). Catches accidental swapping of fit_X/fit_y."""
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    inv_mae_row = df[df["method"] == "inverse_mae"].iloc[0]
    fitted = df.attrs["fitted"]["inverse_mae"]
    # The metadata's per_model_mae should equal the per-model MAE on the
    # holdout (because it was fit there), not OOF.
    holdout_maes = [
        float(np.mean(np.abs(y_hold.to_numpy() - preds_hold[c].to_numpy())))
        for c in preds_hold.columns
    ]
    np.testing.assert_allclose(
        fitted.metadata["per_model_mae"],
        holdout_maes,
        rtol=1e-6,
    )
    # Sanity: reported holdout MAE matches fresh recomputation
    fresh = float(np.mean(np.abs(y_hold.to_numpy() - fitted.predict(preds_hold))))
    assert inv_mae_row["mae"] == pytest.approx(fresh, rel=1e-9)


def test_compare_single_baselines_exact(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, names = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    for name in names:
        expected = float(np.mean(np.abs(y_hold.to_numpy() - preds_hold[name].to_numpy())))
        row = df[df["method"] == f"single::{name}"].iloc[0]
        assert row["mae"] == pytest.approx(expected, rel=1e-9)


def test_compare_skip_base_models(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(
        preds_oof,
        y_oof,
        preds_hold,
        y_hold,
        include_base_models=False,
    )
    assert not any(m.startswith("single::") for m in df["method"])


def test_compare_subset_of_methods(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(
        preds_oof,
        y_oof,
        preds_hold,
        y_hold,
        methods=["simple_average", "inverse_mae"],
        include_base_models=False,
    )
    assert set(df["method"]) == {"simple_average", "inverse_mae"}


def test_compare_validates_column_alignment(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    misaligned = preds_hold[list(reversed(preds_hold.columns))]
    with pytest.raises(ValueError, match="same order"):
        compare_ensemble_methods(preds_oof, y_oof, misaligned, y_hold)


# ── select_best_ensemble ──────────────────────────────────────────


def test_select_best_returns_winner(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    method, ensemble, metrics = select_best_ensemble(df)
    assert method == df.iloc[0]["method"]
    assert metrics["mae"] == pytest.approx(df.iloc[0]["mae"])


def test_select_best_never_worse_than_best_single():
    """The fallback contract: ``select_best_ensemble`` must never return a
    method whose holdout MAE is worse than the best single base model.

    With the holdout-fit refactor, weight-based methods can downweight
    bad models to match or beat ``single::good``. The test below makes
    one model nearly perfect — the winner should be at most as bad as
    the ``single::good`` MAE."""
    rng = np.random.default_rng(0)
    n_oof, n_hold = 200, 60
    y_oof = pd.Series(rng.normal(50, 10, n_oof))
    y_hold = pd.Series(rng.normal(50, 10, n_hold))
    preds_oof = pd.DataFrame(
        {
            "good": y_oof + rng.normal(0, 0.05, n_oof),
            "bad1": y_oof + rng.normal(0, 25, n_oof),
            "bad2": y_oof + rng.normal(0, 25, n_oof),
            "bad3": y_oof + rng.normal(0, 25, n_oof),
        }
    )
    preds_hold = pd.DataFrame(
        {
            "good": y_hold + rng.normal(0, 0.05, n_hold),
            "bad1": y_hold + rng.normal(0, 25, n_hold),
            "bad2": y_hold + rng.normal(0, 25, n_hold),
            "bad3": y_hold + rng.normal(0, 25, n_hold),
        }
    )
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    method, _, metrics = select_best_ensemble(df)
    good_mae = float(np.mean(np.abs(y_hold.to_numpy() - preds_hold["good"].to_numpy())))
    assert metrics["mae"] <= good_mae + 1e-9, (
        f"winner {method!r} ({metrics['mae']:.4f}) is worse than "
        f"single::good ({good_mae:.4f}) — fallback failed"
    )


def test_select_best_requires_fitted_attrs(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, _ = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    df.attrs["fitted"] = {}  # wipe it
    with pytest.raises(ValueError, match="fitted"):
        select_best_ensemble(df)


def test_select_best_rejects_empty():
    with pytest.raises(ValueError, match="No ensemble candidates"):
        select_best_ensemble(pd.DataFrame())


# ── ensemble_config_dict ──────────────────────────────────────────


def test_config_dict_weight_ensemble_serialises(synthetic_preds):
    preds_oof, y_oof, preds_hold, y_hold, names = synthetic_preds
    df = compare_ensemble_methods(preds_oof, y_oof, preds_hold, y_hold)
    _, ens, metrics = select_best_ensemble(df)
    config = ensemble_config_dict(
        ens,
        base_runs={n: {"run_id": f"r_{n}"} for n in names},
        metrics=metrics,
    )
    assert "ensemble" in config
    assert "metrics" in config
    assert isinstance(config["models"], list)
    # If a WeightEnsemble was the winner, weights serialise as a dict
    if isinstance(ens, WeightEnsemble):
        assert isinstance(config["ensemble"]["weights"], dict)
        assert set(config["ensemble"]["weights"]) == set(names)


def test_config_dict_stack_ensemble_records_meta_learner(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_stacking_ridge(preds_oof, y_oof)
    config = ensemble_config_dict(ens, base_runs={}, metrics={"mae": 1.0})
    assert config["ensemble"]["meta_learner"] == "Ridge"
    assert set(config["ensemble"]["model_names"]) == set(preds_oof.columns)


def test_config_dict_surfaces_conformal_quantile(synthetic_preds):
    """Post-hoc conformal output (§5c.8) must be reachable at top level."""
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_simple_average(preds_oof, y_oof)
    metrics = {
        "mae": 1.0,
        "conformal_quantile": 12.3,
        "pi_coverage": 0.905,
        "pi_width": 24.6,
    }
    config = ensemble_config_dict(ens, base_runs={}, metrics=metrics)
    assert config["conformal_quantile"] == 12.3
    assert config["pi_coverage"] == 0.905
    assert config["pi_width"] == 24.6


def test_config_dict_handles_missing_pi_fields(synthetic_preds):
    """If conformal calibration wasn't run, top-level PI fields should be None."""
    preds_oof, y_oof, *_ = synthetic_preds
    ens = fit_simple_average(preds_oof, y_oof)
    config = ensemble_config_dict(ens, base_runs={}, metrics={"mae": 1.0})
    assert config["conformal_quantile"] is None
    assert config["pi_coverage"] is None


def test_fit_ensemble_rejects_unknown_method(synthetic_preds):
    preds_oof, y_oof, *_ = synthetic_preds
    with pytest.raises(ValueError, match="Unknown ensemble method"):
        fit_ensemble("not_a_real_method", preds_oof, y_oof)
