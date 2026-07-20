"""Tests for scripts/ensemble_method_comparison.py scope semantics."""

from __future__ import annotations

import pytest
from scripts.ensemble_method_comparison import _runs_from_config


def _config(all_candidates_fresh: bool = False) -> dict:
    return {
        "ensemble": {"weights": {"m0": 0.7, "m1": 0.0}},
        "artifact_generation": {
            "mode": "bootstrap_reselection" if all_candidates_fresh else "steady_state",
            "fresh_run_names": ["m0", "m1"] if all_candidates_fresh else ["m0"],
            "all_candidates_fresh": all_candidates_fresh,
        },
        "models": [
            {
                "name": "m0",
                "run_id": "run0",
                "model_type": "Ridge",
                "feature_version": "fv",
                "config": {},
            },
            {
                "name": "m1",
                "run_id": "run1",
                "model_type": "LGBMRegressor",
                "feature_version": "fv",
                "config": {},
            },
        ],
    }


def test_comparison_scope_production_uses_positive_weight_members_only():
    runs = _runs_from_config(_config(), "production")

    assert [run.name for run in runs] == ["m0"]


def test_comparison_scope_all_candidates_warns_when_not_fresh():
    with pytest.warns(UserWarning, match="stale sidelined"):
        runs = _runs_from_config(_config(all_candidates_fresh=False), "all-candidates")

    assert {run.name for run in runs} == {"m0", "m1"}


def test_comparison_scope_all_candidates_allows_bootstrap_reselection_artifacts():
    runs = _runs_from_config(_config(all_candidates_fresh=True), "all-candidates")

    assert {run.name for run in runs} == {"m0", "m1"}
