"""Tests for deploy/price_inference.py."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from energy_forecasting.deploy import price_inference as pi


def _merged_frame(start="2026-07-05 00:00", hours=72) -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame(
        {
            "target_price": np.arange(hours, dtype=float),
            "carbon_eur_per_ton": np.linspace(70, 75, hours),
        },
        index=idx,
    )


def test_default_delivery_date_uses_berlin_calendar():
    now = datetime(2026, 7, 6, 22, 30, tzinfo=timezone.utc)

    assert pi._default_delivery_date(now) == date(2026, 7, 8)


def test_extend_uses_requested_delivery_date_not_last_merged_date():
    df = _merged_frame(start="2026-07-05 00:00", hours=72)

    extended, forecast_idx = pi._extend_to_forecast_date(df, forecast_date=date(2026, 7, 7))

    assert forecast_idx[0] == pd.Timestamp("2026-07-07 00:00")
    assert forecast_idx[-1] == pd.Timestamp("2026-07-07 23:00")
    assert len(forecast_idx) == 24
    assert extended.index[-1] == pd.Timestamp("2026-07-07 23:00")


def test_extend_targets_earlier_date_when_merged_already_has_d2():
    """merged.parquet may already contain D+2 rows (prices published early),
    but forecast_date should still anchor to the explicitly requested date."""
    # Frame covers Jul 6, 7, 8 — merged already has Jul 8 data.
    df = _merged_frame(start="2026-07-06 00:00", hours=72)

    extended, forecast_idx = pi._extend_to_forecast_date(df, forecast_date=date(2026, 7, 7))

    assert forecast_idx[0] == pd.Timestamp("2026-07-07 00:00")
    assert forecast_idx[-1] == pd.Timestamp("2026-07-07 23:00")
    assert len(forecast_idx) == 24
    # Target price must be masked for the delivery date regardless of pre-existing values.
    assert extended.loc[forecast_idx, "target_price"].isna().all()
    # Features must still be present.
    assert extended.loc[forecast_idx, "carbon_eur_per_ton"].notna().all()


def test_extend_masks_existing_delivery_target_values():
    df = _merged_frame(start="2026-07-05 00:00", hours=72)

    extended, forecast_idx = pi._extend_to_forecast_date(df, forecast_date=date(2026, 7, 7))

    assert extended.loc[forecast_idx, "target_price"].isna().all()
    assert extended.loc[forecast_idx, "carbon_eur_per_ton"].notna().all()


def test_extend_appends_missing_delivery_rows():
    df = _merged_frame(start="2026-07-05 00:00", hours=48)

    extended, forecast_idx = pi._extend_to_forecast_date(df, forecast_date=date(2026, 7, 7))

    assert len(extended) == 72
    assert forecast_idx[0] == pd.Timestamp("2026-07-07 00:00")
    assert extended.loc[forecast_idx, "target_price"].isna().all()
    assert extended.loc[forecast_idx, "carbon_eur_per_ton"].notna().all()


def test_build_feature_matrices_always_recomputes(monkeypatch):
    calls = []

    def fake_engineer(extended_df, d1_index, feature_version, ds_name):
        calls.append((feature_version, ds_name))
        return pd.DataFrame({"x": [1.0] * len(d1_index)}, index=d1_index)

    monkeypatch.setattr(pi, "_engineer_features_for_version", fake_engineer)
    df = _merged_frame()
    idx = pd.date_range("2026-07-07 00:00", periods=24, freq="h")

    result = pi._build_feature_matrices(df, idx, {"fs_shap_top90"})

    assert calls == [("fs_shap_top90", "price_fs_shap_top90")]
    assert list(result) == ["fs_shap_top90"]
    assert result["fs_shap_top90"].shape == (24, 1)


def test_trained_feature_columns_falls_back_to_model_json(tmp_path, monkeypatch):
    cols = {"custom_version": ["a", "__index_level_0__", "b"]}
    (tmp_path / "price_feature_cols.json").write_text(__import__("json").dumps(cols))
    monkeypatch.setattr(pi, "MODELS_DIR", tmp_path)

    assert pi._trained_feature_columns("custom_version", "price_custom_version") == ["a", "b"]


def test_forecast_builder_applied_after_d1_extension(monkeypatch):
    """Strict forecast column construction must see the D+1 rows."""
    builder_calls: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    d1_index = pd.date_range("2026-07-08 00:00", periods=24, freq="h")

    def fake_builder(df: pd.DataFrame, *, strict_index=None):
        builder_calls.append((df.index.copy(), strict_index.copy()))
        return df

    monkeypatch.setattr(pi, "_extend_to_forecast_date", lambda df, forecast_date: (
        pd.concat([df, pd.DataFrame({"target_price": [np.nan] * 24}, index=d1_index)]),
        d1_index,
    ))
    monkeypatch.setattr("energy_forecasting.features.forecast_inputs.build_forecast_columns", fake_builder)
    monkeypatch.setattr(pi, "_build_feature_matrices", lambda *a, **kw: {})
    monkeypatch.setattr(pi, "production_model_names", lambda cfg: [])
    monkeypatch.setattr(pi, "load_ensemble_config", lambda: {"models": [], "ensemble": {"weights": {}}})

    df = _merged_frame(start="2026-07-05 00:00", hours=72)
    monkeypatch.setattr(pd, "read_parquet", lambda path: df)

    try:
        pi.run_price_inference()
    except RuntimeError:
        pass

    assert builder_calls, "forecast builder was never called"
    call_index, strict_index = builder_calls[-1]
    assert d1_index[0] in call_index
    pd.testing.assert_index_equal(strict_index, d1_index)


def test_engineer_features_raises_for_missing_trained_columns(monkeypatch):
    idx = pd.date_range("2026-07-07 00:00", periods=24, freq="h")
    df = pd.DataFrame({"target_price": np.nan}, index=idx)

    monkeypatch.setattr(pi, "_trained_feature_columns", lambda feature_version, ds_name: ["missing_feature"])

    def fake_eng(extended_df, feature_list, validate=False):
        return pd.DataFrame({"other_feature": [1.0] * len(extended_df)}, index=extended_df.index)

    monkeypatch.setattr("energy_forecasting.features.engine.engineer_features", fake_eng)

    try:
        pi._engineer_features_for_version(df, idx, "custom_version", "price_custom_version")
    except KeyError as exc:
        assert "missing_feature" in str(exc)
    else:
        raise AssertionError("Expected missing trained feature columns to raise")


def test_trained_feature_columns_rejects_stale_prog_schema(tmp_path, monkeypatch):
    cols = {"custom_version": ["prog_residual"]}
    (tmp_path / "price_feature_cols.json").write_text(__import__("json").dumps(cols))
    monkeypatch.setattr(pi, "MODELS_DIR", tmp_path)

    try:
        pi._trained_feature_columns("custom_version", "price_custom_version")
    except ValueError as exc:
        assert "Unsafe trained price feature schema" in str(exc)
    else:
        raise AssertionError("Expected stale prog_* schema to fail")
