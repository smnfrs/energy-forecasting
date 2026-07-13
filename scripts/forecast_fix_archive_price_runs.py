"""Preserve and archive pre-forecast-contract price MLflow runs.

This is the Phase 2a helper for docs/forecast_fix_retrain_plan.md. It exports
small, reviewable summaries before optionally tagging the old price runs as
archived/pre-contract. It intentionally does not delete artifacts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

from energy_forecasting.config import MLFLOW_TRACKING_URI, MODELS_DIR
from energy_forecasting.config.modeling import EXPERIMENTS

PRICE_EXPERIMENTS = [
    "price_feature_selection",
    "price_model_training",
    "price_production",
]
ARCHIVE_REASON = "pre-forecast-contract; leaky/non-comparable"
PRE_CONTRACT = "prog_leaky"
DEFAULT_OUT = Path("docs/archive/price_pre_forecast_contract")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_cols(runs: pd.DataFrame) -> list[str]:
    preferred = [
        "run_id",
        "experiment_id",
        "status",
        "start_time",
        "end_time",
        "metrics.mae",
        "metrics.cv_mae",
        "metrics.rmse",
        "metrics.cv_rmse",
        "metrics.pi_coverage",
        "tags.stage",
        "tags.feature_version",
        "tags.feature_contract",
        "tags.model_class",
        "tags.target",
        "tags.region",
        "tags.archived",
        "tags.archive_reason",
        "params.model_type",
        "params.scaler",
        "params.target_transform",
        "params.weight_half_life",
    ]
    return [c for c in preferred if c in runs.columns]


def _sort_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if "metrics.mae" in runs.columns:
        return runs.sort_values("metrics.mae", na_position="last")
    if "metrics.cv_mae" in runs.columns:
        return runs.sort_values("metrics.cv_mae", na_position="last")
    if "start_time" in runs.columns:
        return runs.sort_values("start_time", ascending=False)
    return runs


def _search_experiment(key: str) -> tuple[str, pd.DataFrame]:
    path = EXPERIMENTS[key]
    exp = mlflow.get_experiment_by_name(path)
    if exp is None:
        return path, pd.DataFrame()
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    return path, runs


def _write_experiment_exports(key: str, out_dir: Path, top_n: int) -> dict:
    path, runs = _search_experiment(key)
    cols = _safe_cols(runs) if not runs.empty else []
    summary = {
        "registry_key": key,
        "display_path": path,
        "run_count": int(len(runs)),
        "exported_at": _now(),
    }
    stem = key

    if runs.empty:
        (out_dir / f"{stem}_summary.md").write_text(
            f"# {path}\n\nNo runs found.\n"
        )
        return summary

    trimmed = _sort_runs(runs[cols].copy())
    trimmed.to_csv(out_dir / f"{stem}_runs.csv", index=False)
    top = trimmed.head(top_n)
    top.to_csv(out_dir / f"{stem}_top_{top_n}.csv", index=False)

    metric_lines = []
    for metric in ["metrics.mae", "metrics.cv_mae", "metrics.rmse", "metrics.cv_rmse"]:
        if metric in runs.columns and runs[metric].notna().any():
            metric_lines.append(
                f"- `{metric}` best: {float(runs[metric].min()):.6f}; "
                f"median: {float(runs[metric].median()):.6f}"
            )

    markdown = [
        f"# {path}",
        "",
        f"- Registry key: `{key}`",
        f"- Runs: {len(runs)}",
        f"- Exported: {_now()}",
        "- Contract disposition: pre-forecast-contract, leaky/non-comparable",
        "",
        "## Metrics",
        "",
        *(metric_lines or ["No metric columns found."]),
        "",
        f"## Top {min(top_n, len(top))} Runs",
        "",
        "```text",
        top.to_string(index=False, max_colwidth=80),
        "```",
        "",
    ]
    (out_dir / f"{stem}_summary.md").write_text("\n".join(markdown))
    return summary


def _write_production_snapshot(out_dir: Path) -> None:
    config_path = MODELS_DIR / "ensemble_config.json"
    feature_cols_path = MODELS_DIR / "price_feature_cols.json"

    snapshot: dict = {
        "exported_at": _now(),
        "contract_disposition": "pre-forecast-contract; leaky/non-comparable",
        "leakage_inflated_holdout_mae": 11.148329224470244,
        "baseline_note": (
            "This MAE was measured with leaky/prognosis-derived prog_* features and "
            "is archived for audit only; it is not a target for the honest forecast_v1 retrain."
        ),
    }

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        snapshot["ensemble"] = cfg.get("ensemble", {})
        snapshot["metrics"] = cfg.get("metrics", {})
        snapshot["conformal_quantile"] = cfg.get("conformal_quantile")
        snapshot["pi_coverage"] = cfg.get("pi_coverage")
        snapshot["models"] = [
            {
                "name": m.get("name"),
                "run_id": m.get("run_id"),
                "model_type": m.get("model_type"),
                "feature_version": m.get("feature_version"),
                "weight": cfg.get("ensemble", {}).get("weights", {}).get(m.get("name"), 0.0),
                "config": m.get("config", {}),
            }
            for m in cfg.get("models", [])
        ]
        (out_dir / "ensemble_config_pre_forecast_contract.json").write_text(
            json.dumps(cfg, indent=2)
        )

    if feature_cols_path.exists():
        feature_cols = json.loads(feature_cols_path.read_text())
        snapshot["price_feature_cols"] = {
            version: len(cols) for version, cols in feature_cols.items()
        }
        (out_dir / "price_feature_cols_pre_forecast_contract.json").write_text(
            json.dumps(feature_cols, indent=2)
        )

    (out_dir / "production_hyperparameters_pre_forecast_contract.json").write_text(
        json.dumps(snapshot, indent=2)
    )


def export(out_dir: Path, top_n: int) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "exported_at": _now(),
        "archive_reason": ARCHIVE_REASON,
        "pre_contract_feature_contract": PRE_CONTRACT,
        "experiments": [],
    }
    for key in PRICE_EXPERIMENTS:
        manifest["experiments"].append(_write_experiment_exports(key, out_dir, top_n))
    _write_production_snapshot(out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _sqlite_path_from_tracking_uri() -> Path:
    uri = str(MLFLOW_TRACKING_URI)
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        raise RuntimeError(f"Expected sqlite MLflow tracking URI, got {uri!r}")
    return Path(uri.removeprefix(prefix))


def archive(out_dir: Path) -> dict:
    """Bulk-tag price runs as archived/pre-contract without deleting artifacts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    db_path = _sqlite_path_from_tracking_uri()
    archive_manifest = {
        "archived_at": _now(),
        "archive_reason": ARCHIVE_REASON,
        "feature_contract": PRE_CONTRACT,
        "method": "sqlite_transaction",
        "experiments": [],
    }

    exp_specs = []
    for key in PRICE_EXPERIMENTS:
        path = EXPERIMENTS[key]
        exp = mlflow.get_experiment_by_name(path)
        if exp is None:
            archive_manifest["experiments"].append(
                {"registry_key": key, "display_path": path, "archived_runs": 0}
            )
            continue
        exp_specs.append((key, path, int(exp.experiment_id)))

    exp_ids = [spec[2] for spec in exp_specs]
    if exp_ids:
        placeholders = ",".join("?" for _ in exp_ids)
        with sqlite3.connect(db_path) as conn:
            conn.execute("BEGIN")
            for tag_key, tag_value in [
                ("archived", "true"),
                ("archive_reason", ARCHIVE_REASON),
                ("feature_contract", PRE_CONTRACT),
            ]:
                conn.execute(
                    f"""
                    INSERT INTO tags(key, value, run_uuid)
                    SELECT ?, ?, run_uuid
                    FROM runs
                    WHERE experiment_id IN ({placeholders})
                    ON CONFLICT(key, run_uuid) DO UPDATE SET value=excluded.value
                    """,
                    [tag_key, tag_value, *exp_ids],
                )
            conn.commit()

    for key, path, exp_id in exp_specs:
        path2, runs = _search_experiment(key)
        run_count = int(len(runs))
        archive_manifest["experiments"].append(
            {"registry_key": key, "display_path": path2, "archived_runs": run_count}
        )
        print(f"Archived {run_count} runs in {path}")

    (out_dir / "archive_manifest.json").write_text(json.dumps(archive_manifest, indent=2))
    return archive_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--archive", action="store_true", help="Tag price runs archived=true and feature_contract=prog_leaky")
    args = parser.parse_args()

    manifest = export(args.out_dir, args.top_n)
    print(json.dumps(manifest, indent=2))
    if args.archive:
        archive_manifest = archive(args.out_dir)
        print(json.dumps(archive_manifest, indent=2))


if __name__ == "__main__":
    main()
