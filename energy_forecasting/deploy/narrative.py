"""AI narrative generation for the Stage 10 forecast story.

Two independent narratives, each with its own cadence and its own Groq call:

- Yearly recap (weekly): plain-English summary of the trailing year's
  generation/load mix and price behaviour, vs. the prior year and vs. the
  same week ~52 weeks back. Facts come from scripts/build_forecast_story_data.py.
- Forecast driver explanation (daily): what tomorrow's gen/load and price
  forecasts look like vs. recent history, and — for price — which real SHAP
  categories are pushing the ensemble's forecast away from its own baseline.
  Facts come from deploy/data/price_shap.json (written by run_price_inference,
  see energy_forecasting/deploy/shap_attribution.py) plus the other already-
  published deploy/data/*.json files.

Facts assembly never touches the network and never raises. Only the Groq call
(prose generation) can fail, and it degrades independently: any failure —
missing key, HTTP error, timeout, malformed JSON, missing output keys — is
treated uniformly as unavailable, logged as a warning, never raised. The facts
payload is always persisted, even when the LLM call fails, so it's usable
standalone.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

from energy_forecasting.config import DEPLOY_DATA_DIR, DEPLOY_DIR

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
_TIMEOUT_S = 20
_MAX_TOKENS = 700
_TEMPERATURE = 0.4

FORECAST_STORY_DATA_DIR = DEPLOY_DIR / "story" / "forecast" / "data"
FACTS_YEARLY_PATH = FORECAST_STORY_DATA_DIR / "facts_yearly.json"
NARRATIVE_YEARLY_PATH = FORECAST_STORY_DATA_DIR / "narrative_yearly.json"
NARRATIVE_FORECAST_PATH = DEPLOY_DATA_DIR / "narrative_forecast.json"
PRICE_SHAP_PATH = DEPLOY_DATA_DIR / "price_shap.json"
GEN_LOAD_DATA_DIR = DEPLOY_DATA_DIR / "gen_load"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"narrative: could not parse {path}")
        return None


def _call_groq(
    system_prompt: str,
    user_content: str,
    model: str,
    required_keys: tuple[str, ...],
) -> tuple[dict | None, str | None]:
    """Call Groq's chat completions endpoint; return (parsed_json, reason).

    reason is None on success, else one of "no_api_key", "api_error",
    "malformed_response". Never raises.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, skipping narrative generation")
        return None, "no_api_key"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": _TEMPERATURE,
        "max_tokens": _MAX_TOKENS,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:
        logger.warning(f"narrative: Groq call failed — {exc}")
        return None, "api_error"

    missing = [k for k in required_keys if k not in parsed]
    if missing:
        logger.warning(f"narrative: Groq response missing keys {missing}")
        return None, "malformed_response"

    return parsed, None


# ── Yearly recap (weekly, Groq call #1) ───────────────────────────────


def _condense_yearly_block(block: dict) -> dict:
    gl = block["gen_load"]
    price = block["price"]
    return {
        "fuel_mix_pct": gl["fuel_mix_pct"],
        "imports_pct_of_domestic_gen": gl["imports_pct_of_domestic_gen"],
        "exports_pct_of_domestic_gen": gl["exports_pct_of_domestic_gen"],
        "mean_price_eur_mwh": price["mean_price"],
        "most_expensive_hour": price["most_expensive_hour"],
        "most_expensive_hour_avg_price": price["most_expensive_hour_avg_price"],
        "least_expensive_hour": price["least_expensive_hour"],
        "least_expensive_hour_avg_price": price["least_expensive_hour_avg_price"],
        "negative_price_hours": price["negative_price_hours"],
    }


def assemble_yearly_facts() -> dict | None:
    """Condense scripts/build_forecast_story_data.py's output to summary scalars.

    Drops the raw daily/hourly series — only the aggregate numbers needed for
    the narrative are kept, both to bound the prompt size and to keep the
    persisted facts payload small.
    """
    raw = _load_json(FACTS_YEARLY_PATH)
    if raw is None:
        return None
    return {
        "generated_through": raw["generated_through"],
        "current_year": _condense_yearly_block(raw["current_year"]),
        "prior_year": _condense_yearly_block(raw["prior_year"]),
        "last_7_days": _condense_yearly_block(raw["last_7_days"]),
        "same_7_days_prior_year": _condense_yearly_block(raw["same_7_days_prior_year"]),
    }


_YEARLY_SYSTEM_PROMPT = """You are an energy-market analyst writing two short factual \
paragraphs for a public dashboard about the German (DE-LU) day-ahead electricity market.

You will be given trailing-365-day statistics for generation mix, load, cross-border \
imports/exports, and price, alongside the same statistics for the prior year and for the \
last 7 days vs. the same 7 days one year earlier.

Rules:
- State only the numbers you are given. Never invent a number, date, or event.
- Write in plain, neutral English for a general audience — no jargon without a one-clause \
explanation.
- Two fields only. "gen_load_yearly_summary": 2-4 sentences on the generation mix and load \
trend, comparing the trailing year to the prior year and the last week to the same week a \
year ago. "price_yearly_summary": 2-4 sentences on price behaviour over the same comparisons \
(mean price, cheapest/most expensive hours of day, negative-price hours).
- Respond with strict JSON: {"gen_load_yearly_summary": "...", "price_yearly_summary": "..."}
"""


def generate_yearly_narrative(model: str = DEFAULT_MODEL) -> dict:
    """Generate the weekly yearly-recap narrative; write narrative_yearly.json.

    Never raises. Always persists a result, with status "ok" or "unavailable".
    """
    generated_at = _now_utc()
    facts = assemble_yearly_facts()

    if facts is None:
        result = {
            "generated_at": generated_at,
            "model": model,
            "status": "unavailable",
            "reason": "facts_error",
            "gen_load_yearly_summary": None,
            "price_yearly_summary": None,
            "facts": None,
        }
        FORECAST_STORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        NARRATIVE_YEARLY_PATH.write_text(json.dumps(result, indent=2))
        logger.warning("generate_yearly_narrative: no facts available (run build_forecast_story_data.py first)")
        return result

    parsed, reason = _call_groq(
        _YEARLY_SYSTEM_PROMPT,
        json.dumps(facts),
        model,
        required_keys=("gen_load_yearly_summary", "price_yearly_summary"),
    )

    result = {
        "generated_at": generated_at,
        "model": model,
        "status": "ok" if parsed else "unavailable",
        "reason": reason,
        "gen_load_yearly_summary": parsed.get("gen_load_yearly_summary") if parsed else None,
        "price_yearly_summary": parsed.get("price_yearly_summary") if parsed else None,
        "facts": facts,
    }
    FORECAST_STORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    NARRATIVE_YEARLY_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Written {NARRATIVE_YEARLY_PATH} (status={result['status']})")
    return result


# ── Forecast driver explanation (daily, Groq call #2) ─────────────────


def _series_stats(entries: list[dict], key: str = "forecast") -> dict | None:
    values = [e[key] for e in entries if e.get(key) is not None]
    if not values:
        return None
    return {"mean": round(sum(values) / len(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}


def _recent_actual_stats(days: list[dict]) -> dict | None:
    values = [v for d in days for v in d.get("values", d.get("prices", []))]
    if not values:
        return None
    return {"mean": round(sum(values) / len(values), 1), "min": round(min(values), 1), "max": round(max(values), 1)}


def _category_attributions(shap: dict) -> list[dict]:
    """Per-category signed mean contribution (EUR/MWh) + % of total absolute contribution,
    ranked by magnitude, read directly from price_shap.json — no recomputation."""
    contributions = shap.get("category_contributions", {})
    totals = {cat: sum(vals) / len(vals) for cat, vals in contributions.items() if vals}
    abs_sum = sum(abs(v) for v in totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {
            "category": cat,
            "mean_contribution_eur_mwh": round(mean_v, 2),
            "pct_of_total_abs_contribution": round(abs(mean_v) / abs_sum * 100, 1),
        }
        for cat, mean_v in ranked
    ]


def assemble_forecast_facts() -> dict | None:
    """Read already-published JSON only — no model loading or feature engineering."""
    price_forecast = _load_json(DEPLOY_DATA_DIR / "price_forecast.json")
    if price_forecast is None:
        return None

    actuals = _load_json(DEPLOY_DATA_DIR / "actuals.json")
    shap = _load_json(PRICE_SHAP_PATH)
    model_metadata = _load_json(DEPLOY_DATA_DIR / "model_metadata.json")
    errors_summary = _load_json(DEPLOY_DATA_DIR / "errors_summary.json")
    gen_load_actuals = _load_json(DEPLOY_DATA_DIR / "gen_load_actuals.json") or {}

    gen_load_forecast_stats = {}
    for target in ("wind_onshore", "wind_offshore", "solar", "load"):
        gl = _load_json(GEN_LOAD_DATA_DIR / f"{target}_national.json")
        if gl:
            gen_load_forecast_stats[target] = _series_stats(gl["forecasts"])

    gen_load_recent_actual_stats = {
        target: _recent_actual_stats(payload["days"])
        for target, payload in gen_load_actuals.items()
        if payload.get("days")
    }

    recent_actual_price = None
    if actuals and actuals.get("days"):
        recent_actual_price = _recent_actual_stats(actuals["days"][-7:])

    facts = {
        "delivery_date": price_forecast["forecasts"][0]["timestamp"][:10],
        "price_forecast": _series_stats(price_forecast["forecasts"]),
        "recent_actual_price_7d": recent_actual_price,
        "gen_load_forecast": gen_load_forecast_stats,
        "gen_load_recent_actual_7d": gen_load_recent_actual_stats,
        "ensemble": {
            "models": model_metadata.get("models") if model_metadata else None,
            "holdout_mae": model_metadata.get("holdout_mae") if model_metadata else None,
        }
        if model_metadata
        else None,
        "recent_price_mae_7d": (
            round(sum(errors_summary["mae"][-7:]) / len(errors_summary["mae"][-7:]), 2)
            if errors_summary and errors_summary.get("mae")
            else None
        ),
        "category_attributions": _category_attributions(shap) if shap else None,
        "shap_base_value_eur_mwh": shap.get("base_value") if shap else None,
    }
    return facts


_FORECAST_SYSTEM_PROMPT = """You are an energy-market analyst writing two short factual \
paragraphs for a public dashboard about tomorrow's German (DE-LU) day-ahead electricity \
forecast.

You will be given tomorrow's generation/load and price forecasts, recent actuals for \
comparison, the production ensemble's recent accuracy, and — for price — a real SHAP \
(SHapley Additive exPlanations) attribution: a mathematical decomposition of the ensemble \
model's own prediction into signed contributions per category (gas, carbon, oil, neighbour \
prices, wind, solar, residual generation, conventional generation, cross-border flows, load, \
price momentum, calendar), relative to the model's own baseline.

Rules:
- State only the numbers you are given. Never invent a number, date, or event.
- The SHAP attribution explains this specific model's own reasoning, not verified real-world \
market causality — include one explicit sentence making this distinction clear.
- Two fields only. "gen_load_forecast_note": 2-3 sentences comparing tomorrow's gen/load \
forecast to the last 7 days, flagging anything extreme (near the recent min/max). \
"price_driver_explanation": 2-4 sentences stating the top 2-3 SHAP categories by magnitude \
and their signed contribution (EUR/MWh) toward tomorrow's price forecast, ending with the \
epistemic-humility sentence above.
- Respond with strict JSON: {"gen_load_forecast_note": "...", "price_driver_explanation": "..."}
"""


def generate_forecast_narrative(model: str = DEFAULT_MODEL) -> dict:
    """Generate the daily forecast-driver narrative; write narrative_forecast.json.

    Never raises. Always persists a result, with status "ok" or "unavailable".
    """
    generated_at = _now_utc()
    facts = assemble_forecast_facts()

    if facts is None:
        result = {
            "generated_at": generated_at,
            "delivery_date": None,
            "model": model,
            "status": "unavailable",
            "reason": "facts_error",
            "gen_load_forecast_note": None,
            "price_driver_explanation": None,
            "facts": None,
        }
        DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        NARRATIVE_FORECAST_PATH.write_text(json.dumps(result, indent=2))
        logger.warning("generate_forecast_narrative: no facts available (run deploy forecast first)")
        return result

    parsed, reason = _call_groq(
        _FORECAST_SYSTEM_PROMPT,
        json.dumps(facts),
        model,
        required_keys=("gen_load_forecast_note", "price_driver_explanation"),
    )

    result = {
        "generated_at": generated_at,
        "delivery_date": facts["delivery_date"],
        "model": model,
        "status": "ok" if parsed else "unavailable",
        "reason": reason,
        "gen_load_forecast_note": parsed.get("gen_load_forecast_note") if parsed else None,
        "price_driver_explanation": parsed.get("price_driver_explanation") if parsed else None,
        "facts": facts,
    }
    DEPLOY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    NARRATIVE_FORECAST_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Written {NARRATIVE_FORECAST_PATH} (status={result['status']})")
    return result
