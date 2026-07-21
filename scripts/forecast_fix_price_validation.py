#!/usr/bin/env python3
"""Post-remediation price validation for the forecast-fix coverage plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from energy_forecasting.config import MODELS_DIR
from energy_forecasting.data.io import load_parquet
from energy_forecasting.features.forecast_inputs import build_forecast_columns
from energy_forecasting.modeling.baselines import naive_seasonal_7d
from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float((aligned["y_true"] - aligned["y_pred"]).abs().mean())


def _download_holdout_predictions(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    ensure_mlflow_tracking()
    client = mlflow.MlflowClient()
    frames: dict[str, pd.DataFrame] = {}
    for entry in config["models"]:
        run_id = entry["run_id"]
        model_name = entry["name"]
        artifact_dir = Path(client.download_artifacts(run_id, "predictions"))
        holdout = pd.read_parquet(artifact_dir / "holdout_predictions.parquet")
        frames[model_name] = holdout

    names = list(frames)
    common_index = frames[names[0]].index
    for name in names[1:]:
        common_index = common_index.intersection(frames[name].index)

    preds = pd.DataFrame(
        {name: frames[name].loc[common_index, "y_pred"] for name in names},
        index=common_index,
    )
    y_true = frames[names[0]].loc[common_index, "y_true"]
    return preds, y_true


def _ensemble_prediction(config: dict[str, Any], preds: pd.DataFrame) -> pd.Series:
    weights = config["ensemble"]["weights"]
    result = pd.Series(0.0, index=preds.index, name="ensemble_pred")
    for name, weight in weights.items():
        if name not in preds.columns:
            raise KeyError(f"Missing prediction column for weighted model: {name}")
        result = result + float(weight) * preds[name]
    return result


def _slice_metrics(
    name: str,
    mask: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    baseline: pd.Series,
) -> dict[str, Any]:
    mask = mask.reindex(y_true.index).fillna(False).astype(bool)
    yt = y_true[mask]
    yp = y_pred[mask]
    bs = baseline[mask]
    model_mae = _mae(yt, yp)
    baseline_mae = _mae(yt, bs)
    skill = float(1.0 - model_mae / baseline_mae) if baseline_mae and not np.isnan(baseline_mae) else None
    return {
        "name": name,
        "rows": int(mask.sum()),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_skill_vs_baseline": skill,
    }


def validate(
    config_path: Path = MODELS_DIR / "ensemble_config.json",
    merged_path: Path = Path("data/processed/merged.parquet"),
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    preds, y_true = _download_holdout_predictions(config)
    y_pred = _ensemble_prediction(config, preds)

    raw = load_parquet(merged_path)
    with_forecasts = build_forecast_columns(raw)
    prices = raw["target_price"]
    baseline_full = naive_seasonal_7d(prices, n_weeks=4)
    baseline = baseline_full.reindex(y_true.index)

    holdout_features = with_forecasts.reindex(y_true.index)
    solar = holdout_features["forecast_gen_solar"]
    residual_ramp = holdout_features["forecast_residual_load"].diff().abs()

    slices = [
        _slice_metrics("all_holdout", pd.Series(True, index=y_true.index), y_true, y_pred, baseline),
        _slice_metrics(
            "high_solar_top_decile",
            solar >= solar.quantile(0.90),
            y_true,
            y_pred,
            baseline,
        ),
        _slice_metrics("negative_price_hours", y_true < 0, y_true, y_pred, baseline),
        _slice_metrics(
            "residual_load_ramp_top_decile",
            residual_ramp >= residual_ramp.quantile(0.90),
            y_true,
            y_pred,
            baseline,
        ),
    ]

    return {
        "config_path": str(config_path),
        "merged_path": str(merged_path),
        "ensemble_method": config["ensemble"].get("method"),
        "config_metrics": config.get("metrics", {}),
        "nonzero_weights": {
            name: weight
            for name, weight in config["ensemble"]["weights"].items()
            if abs(float(weight)) > 1e-9
        },
        "holdout_start": y_true.index.min().isoformat(),
        "holdout_end": y_true.index.max().isoformat(),
        "holdout_rows": int(len(y_true)),
        "slices": slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev-notes/forecast_fix/forecast_fix_price_validation_20260715.json"),
    )
    args = parser.parse_args()

    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
