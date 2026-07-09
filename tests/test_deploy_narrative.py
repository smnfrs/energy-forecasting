"""Tests for deploy/narrative.py — facts assembly + Groq call failure contract."""

import json

import pytest
import responses

from energy_forecasting.deploy import narrative as nar


def test_call_groq_no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    parsed, reason = nar._call_groq("system", "user", "some-model", required_keys=("a",))
    assert parsed is None
    assert reason == "no_api_key"


@responses.activate
def test_call_groq_http_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    responses.add(responses.POST, nar.GROQ_URL, status=500)
    parsed, reason = nar._call_groq("system", "user", "some-model", required_keys=("a",))
    assert parsed is None
    assert reason == "api_error"


@responses.activate
def test_call_groq_malformed_json_body(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    responses.add(
        responses.POST, nar.GROQ_URL, status=200,
        json={"choices": [{"message": {"content": "not valid json"}}]},
    )
    parsed, reason = nar._call_groq("system", "user", "some-model", required_keys=("a",))
    assert parsed is None
    assert reason == "api_error"


@responses.activate
def test_call_groq_missing_required_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    responses.add(
        responses.POST, nar.GROQ_URL, status=200,
        json={"choices": [{"message": {"content": json.dumps({"other": "x"})}}]},
    )
    parsed, reason = nar._call_groq("system", "user", "some-model", required_keys=("a",))
    assert parsed is None
    assert reason == "malformed_response"


@responses.activate
def test_call_groq_success(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    responses.add(
        responses.POST, nar.GROQ_URL, status=200,
        json={"choices": [{"message": {"content": json.dumps({"a": "hello"})}}]},
    )
    parsed, reason = nar._call_groq("system", "user", "some-model", required_keys=("a",))
    assert parsed == {"a": "hello"}
    assert reason is None


def test_generate_yearly_narrative_no_facts_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(nar, "FACTS_YEARLY_PATH", tmp_path / "facts_yearly.json")
    monkeypatch.setattr(nar, "NARRATIVE_YEARLY_PATH", tmp_path / "narrative_yearly.json")
    monkeypatch.setattr(nar, "FORECAST_STORY_DATA_DIR", tmp_path)

    result = nar.generate_yearly_narrative()

    assert result["status"] == "unavailable"
    assert result["reason"] == "facts_error"
    assert json.loads((tmp_path / "narrative_yearly.json").read_text())["status"] == "unavailable"


def test_generate_forecast_narrative_no_facts_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(nar, "DEPLOY_DATA_DIR", tmp_path)
    monkeypatch.setattr(nar, "NARRATIVE_FORECAST_PATH", tmp_path / "narrative_forecast.json")
    monkeypatch.setattr(nar, "PRICE_SHAP_PATH", tmp_path / "price_shap.json")
    monkeypatch.setattr(nar, "GEN_LOAD_DATA_DIR", tmp_path / "gen_load")

    result = nar.generate_forecast_narrative()

    assert result["status"] == "unavailable"
    assert result["reason"] == "facts_error"
    assert json.loads((tmp_path / "narrative_forecast.json").read_text())["status"] == "unavailable"


def test_category_attributions_ranks_by_magnitude():
    shap = {
        "category_contributions": {
            "gas": [1.0, 1.0],
            "wind": [-5.0, -5.0],
            "carbon": [0.5, 0.5],
        }
    }
    ranked = nar._category_attributions(shap)
    assert [r["category"] for r in ranked] == ["wind", "gas", "carbon"]
    assert ranked[0]["mean_contribution_eur_mwh"] == -5.0
    assert pytest.approx(sum(r["pct_of_total_abs_contribution"] for r in ranked), abs=0.1) == 100.0
