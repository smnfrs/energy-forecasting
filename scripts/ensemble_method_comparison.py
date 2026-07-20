#!/usr/bin/env python3
"""Diagnostic comparison of price ensemble methods.

This script is intentionally not part of production ensemble construction. It
loads existing prediction artifacts and compares the research method registry on
OOF-fit / holdout-eval semantics.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from energy_forecasting.deploy.model_store import ENSEMBLE_CONFIG_PATH, production_model_names
from energy_forecasting.modeling.ensemble import compare_ensemble_methods, stack_model_predictions
from energy_forecasting.modeling.price import _fetch_predictions


@dataclass
class _Run:
    name: str
    run_id: str
    model_type: str
    feature_version: str
    config: dict[str, Any]


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _runs_from_config(config: dict[str, Any], scope: str) -> list[_Run]:
    entries = list(config.get("models", []))
    if scope == "production":
        names = set(production_model_names(config))
        entries = [entry for entry in entries if entry["name"] in names]
    elif scope == "all-candidates":
        generation = config.get("artifact_generation", {})
        if not generation.get("all_candidates_fresh", False):
            warnings.warn(
                "--scope all-candidates requested, but config is not marked as a "
                "fresh bootstrap/reselection artifact set. Results may mix fresh "
                "production members with stale sidelined candidates.",
                stacklevel=2,
            )
    else:
        raise ValueError(f"Unknown scope: {scope}")

    return [
        _Run(
            name=entry["name"],
            run_id=entry["run_id"],
            model_type=entry.get("model_type", entry["name"]),
            feature_version=entry.get("feature_version", "unknown"),
            config=entry.get("config", {}),
        )
        for entry in entries
    ]


def _runs_from_ids(run_ids: list[str]) -> list[_Run]:
    return [
        _Run(
            name=f"run_{idx}_{run_id[:8]}",
            run_id=run_id,
            model_type="unknown",
            feature_version="unknown",
            config={},
        )
        for idx, run_id in enumerate(run_ids)
    ]


def _parse_methods(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    return methods or None


def run_comparison(
    *,
    config_path: Path,
    scope: str,
    run_ids: list[str] | None,
    methods: list[str] | None,
) -> pd.DataFrame:
    if run_ids:
        runs = _runs_from_ids(run_ids)
    else:
        config = _load_config(config_path)
        runs = _runs_from_config(config, scope)
    if not runs:
        raise RuntimeError("No model runs selected for ensemble comparison")

    preds_oof, y_oof, preds_holdout, y_holdout, alignment = stack_model_predictions(
        runs,
        prediction_loader=_fetch_predictions,
    )
    table = compare_ensemble_methods(
        preds_oof,
        y_oof,
        preds_holdout,
        y_holdout,
        methods=methods,
    )
    table.attrs["alignment_metadata"] = alignment
    return table


def _write_markdown(table: pd.DataFrame, path: Path) -> None:
    path.write_text(table.to_markdown(index=False) + "\n")


def _log_to_mlflow(table: pd.DataFrame) -> None:
    import mlflow
    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking

    ensure_mlflow_tracking()
    mlflow.set_experiment("price/ensemble_research")
    with mlflow.start_run(run_name="ensemble_method_comparison"):
        best = table.sort_values("mae").iloc[0]
        mlflow.log_param("best_method", str(best["method"]))
        mlflow.log_metric("best_mae", float(best["mae"]))
        mlflow.log_dict(table.to_dict(orient="records"), "ensemble_method_comparison.json")
        alignment = table.attrs.get("alignment_metadata")
        if alignment:
            mlflow.log_dict(alignment, "ensemble_alignment.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ENSEMBLE_CONFIG_PATH)
    parser.add_argument(
        "--scope",
        choices=("production", "all-candidates"),
        default="production",
        help="Use positive-weight production members or every configured candidate.",
    )
    parser.add_argument("--run-ids", nargs="*", help="Explicit MLflow run IDs to compare.")
    parser.add_argument("--methods", help="Comma-separated method subset.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--mlflow", action="store_true", help="Log comparison to MLflow.")
    args = parser.parse_args()

    table = run_comparison(
        config_path=args.config,
        scope=args.scope,
        run_ids=args.run_ids,
        methods=_parse_methods(args.methods),
    )
    print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
    if args.markdown:
        _write_markdown(table, args.markdown)
    if args.mlflow:
        _log_to_mlflow(table)


if __name__ == "__main__":
    main()
