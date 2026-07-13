"""Audit regenerated price datasets for the forecast_v1 contract."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from energy_forecasting.config import PROCESSED_DATA_DIR
from energy_forecasting.features.forecast_inputs import (
    FORECAST_COLUMNS,
    _derive_from_actuals,
    _derive_from_artifacts,
    _derive_from_smard,
    _load_artifact_layer,
    build_forecast_columns,
)
from energy_forecasting.modeling.datasets import DATASET_DIR

OUT_DIR = Path("docs/archive/price_pre_forecast_contract")
BANNED_TOKENS = ("prog_", "pct_prog_", "prognostiziert")


def _source_labels(df: pd.DataFrame) -> pd.Series:
    artifact = _load_artifact_layer(df.index)
    own_layer = _derive_from_artifacts(artifact)
    smard_layer = _derive_from_smard(df)
    actual_layer = _derive_from_actuals(df)

    own_mask = artifact.notna().all(axis=1)
    smard_mask = smard_layer.notna().all(axis=1)
    actual_mask = actual_layer.notna().all(axis=1)

    labels = pd.Series("missing", index=df.index, dtype="object")
    labels.loc[actual_mask] = "actual"
    labels.loc[smard_mask] = "smard"
    labels.loc[own_mask] = "own"
    return labels


def _schema_audit() -> dict:
    datasets = {}
    banned: dict[str, list[str]] = {}
    for path in sorted(DATASET_DIR.glob("price_*.parquet")):
        cols = pq.read_schema(path).names
        hits = [c for c in cols if any(token in c for token in BANNED_TOKENS)]
        datasets[path.name] = {"n_columns": len(cols), "banned_columns": hits}
        if hits:
            banned[path.name] = hits
    return {"datasets": datasets, "banned": banned}


def _plot_boundary(audited: pd.DataFrame, labels: pd.Series, out_path: Path) -> str | None:
    own_idx = labels.index[labels == "own"]
    if len(own_idx) == 0:
        return None
    first_own = own_idx.min()
    start = first_own - pd.Timedelta(days=14)
    end = first_own + pd.Timedelta(days=14)
    window = audited.loc[start:end]
    if window.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window.index, window["forecast_gen_total"], linewidth=1.2, label="forecast_gen_total")
    ax.axvline(first_own, color="black", linestyle="--", linewidth=1, label=f"first own: {first_own:%Y-%m-%d}")
    ax.set_title("forecast_gen_total around own-vs-SMARD boundary")
    ax.set_ylabel("MW")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return first_own.isoformat()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = PROCESSED_DATA_DIR / "merged.parquet"
    raw = pd.read_parquet(merged_path)
    audited = build_forecast_columns(raw)
    labels = _source_labels(raw)

    nan_counts = {col: int(audited[col].isna().sum()) for col in FORECAST_COLUMNS}
    source_counts = labels.value_counts().reindex(["own", "smard", "actual", "missing"], fill_value=0)
    source_by_year = (
        pd.crosstab(labels.index.year, labels)
        .reindex(columns=["own", "smard", "actual", "missing"], fill_value=0)
        .astype(int)
    )

    own_2022_mask = (labels == "own") & (audited.index >= "2022-01-01")
    residual_calc = audited.loc[own_2022_mask, "forecast_load"] - audited.loc[own_2022_mask, "forecast_gen_wind_pv"]
    residual_delta = (audited.loc[own_2022_mask, "forecast_residual_load"] - residual_calc).abs()
    residual_delta_max = float(residual_delta.max()) if len(residual_delta) else None

    raw_prog = "prognostizierter_verbrauch_residuallast"
    prog_compare = {}
    if raw_prog in raw.columns:
        comp = audited.loc[audited.index >= "2022-01-01", ["forecast_residual_load"]].copy()
        comp["prog_residual"] = raw.loc[comp.index, raw_prog]
        diff = (comp["forecast_residual_load"] - comp["prog_residual"]).dropna()
        prog_compare = {
            "rows_compared_2022_plus": int(len(diff)),
            "mean_abs_diff": float(diff.abs().mean()) if len(diff) else None,
            "max_abs_diff": float(diff.abs().max()) if len(diff) else None,
            "rows_different_gt_1mw": int((diff.abs() > 1.0).sum()),
        }

    holdout_start = audited.index.max() - pd.Timedelta(days=90)
    holdout_labels = labels.loc[labels.index > holdout_start]
    holdout_counts = holdout_labels.value_counts().reindex(["own", "smard", "actual", "missing"], fill_value=0)

    schema = _schema_audit()
    boundary_plot = OUT_DIR / "forecast_gen_total_boundary.png"
    first_own = _plot_boundary(audited, labels, boundary_plot)

    report = {
        "merged_path": str(merged_path),
        "rows": int(len(audited)),
        "forecast_columns": list(FORECAST_COLUMNS),
        "forecast_nan_counts": nan_counts,
        "forecast_source_counts": {k: int(v) for k, v in source_counts.items()},
        "forecast_source_counts_by_year": {
            str(year): {k: int(v) for k, v in row.items()}
            for year, row in source_by_year.to_dict(orient="index").items()
        },
        "first_own_forecast_timestamp": first_own,
        "boundary_plot": str(boundary_plot),
        "own_2022_plus_residual_identity_max_abs_error": residual_delta_max,
        "prog_residual_comparison": prog_compare,
        "holdout_start_exclusive": holdout_start.isoformat(),
        "holdout_source_counts": {k: int(v) for k, v in holdout_counts.items()},
        "dataset_schema_audit": schema,
    }

    (OUT_DIR / "dataset_audit.json").write_text(json.dumps(report, indent=2))
    source_by_year.to_csv(OUT_DIR / "forecast_source_counts_by_year.csv")

    md = [
        "# Forecast Dataset Audit",
        "",
        f"- Rows: {len(audited):,}",
        f"- Forecast NaNs: {nan_counts}",
        f"- Source counts: {report['forecast_source_counts']}",
        f"- First own forecast timestamp: `{first_own}`",
        f"- Own 2022+ residual identity max abs error: {residual_delta_max:.12g}",
        f"- Holdout source counts: {report['holdout_source_counts']}",
        f"- Dataset schema banned-token hits: {schema['banned'] or {}}",
        "",
        "## Source Counts By Year",
        "",
        "```text",
        source_by_year.to_string(),
        "```",
        "",
        "## prog_residual Comparison",
        "",
        "```json",
        json.dumps(prog_compare, indent=2),
        "```",
        "",
    ]
    (OUT_DIR / "dataset_audit.md").write_text("\n".join(md))

    failures = []
    if any(v != 0 for v in nan_counts.values()):
        failures.append(f"forecast NaNs present: {nan_counts}")
    if schema["banned"]:
        failures.append(f"banned schema tokens present: {schema['banned']}")
    if int(holdout_counts.get("actual", 0)) != 0:
        failures.append(f"actual fallback appears in holdout: {holdout_counts.to_dict()}")
    if residual_delta_max is None or residual_delta_max > 1e-9:
        failures.append(f"own 2022+ forecast residual identity mismatch: {residual_delta_max}")

    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
