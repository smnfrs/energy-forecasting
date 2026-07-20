"""Tests for daily inference orchestration."""

from __future__ import annotations


def test_run_inference_no_price_skips_price_path(monkeypatch):
    from energy_forecasting.deploy import gen_load_inference as gli
    from energy_forecasting.deploy import inference as inf
    from energy_forecasting.deploy import price_inference as pi
    from energy_forecasting.deploy import publish as pub
    from energy_forecasting.deploy import validation as val

    calls: list[str] = []
    gen_load_results = {("load", "DE_NATIONAL"): object()}

    monkeypatch.setattr(gli, "run_gen_load_inference", lambda: calls.append("gen_load") or gen_load_results)
    monkeypatch.setattr(
        gli,
        "update_historical_forecasts",
        lambda results: calls.append("update_historical_forecasts"),
    )
    monkeypatch.setattr(pi, "run_price_inference", lambda: calls.append("price"))
    monkeypatch.setattr(val, "validate_outputs", lambda price, gen_load: calls.append("validate"))
    monkeypatch.setattr(pub, "write_outputs", lambda price, gen_load: calls.append("write_all"))
    monkeypatch.setattr(
        pub,
        "write_gen_load_only_outputs",
        lambda results: calls.append("write_gen_load_only"),
    )

    result = inf.run_inference(skip_update=True, run_price=False)

    assert result == {"price": None, "gen_load": gen_load_results}
    assert calls == ["gen_load", "update_historical_forecasts", "write_gen_load_only"]


def test_run_inference_price_checks_coverage_before_price(monkeypatch):
    from energy_forecasting.deploy import gen_load_inference as gli
    from energy_forecasting.deploy import inference as inf
    from energy_forecasting.deploy import price_inference as pi
    from energy_forecasting.deploy import publish as pub
    from energy_forecasting.deploy import validation as val
    from energy_forecasting.features import forecast_coverage as cov

    calls: list[str] = []
    gen_load_results = {("load", "DE_NATIONAL"): object()}
    price_df = object()

    monkeypatch.setattr(gli, "run_gen_load_inference", lambda: calls.append("gen_load") or gen_load_results)
    monkeypatch.setattr(
        gli,
        "update_historical_forecasts",
        lambda results: calls.append("update_historical_forecasts"),
    )
    monkeypatch.setattr(
        cov,
        "assert_monthly_artifact_coverage",
        lambda: calls.append("coverage") or {"months_checked": 1},
    )
    monkeypatch.setattr(pi, "run_price_inference", lambda: calls.append("price") or price_df)
    monkeypatch.setattr(val, "validate_outputs", lambda price, gen_load: calls.append("validate"))
    monkeypatch.setattr(pub, "write_outputs", lambda price, gen_load: calls.append("write_all"))

    result = inf.run_inference(skip_update=True, run_price=True)

    assert result == {"price": price_df, "gen_load": gen_load_results}
    assert calls == [
        "gen_load",
        "update_historical_forecasts",
        "coverage",
        "price",
        "validate",
        "write_all",
    ]
