"""Tests for MLflow tracking helpers."""

from __future__ import annotations

import pytest

from energy_forecasting.modeling.mlflow_utils import TrackedRun


def test_tracked_run_requires_feature_contract():
    with pytest.raises(ValueError, match="feature_contract"):
        with TrackedRun(
            "price_model_training",
            stage="model_training",
            feature_version="fs_test",
        ):
            raise AssertionError("run should not start")


def test_tracked_run_rejects_blank_feature_contract():
    with pytest.raises(ValueError, match="feature_contract"):
        with TrackedRun(
            "price_model_training",
            stage="model_training",
            feature_version="fs_test",
            feature_contract=" ",
        ):
            raise AssertionError("run should not start")
