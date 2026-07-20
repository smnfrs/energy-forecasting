"""Operational coverage tools for the forecast_v1 remediation.

Subcommands:
  spans     Dry-calculate expected gen/load OOF+holdout export spans.
  coverage  Report current artifact source coverage and price holdout mix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from energy_forecasting.config import PROCESSED_DATA_DIR
from energy_forecasting.config.modeling import (
    GEN_LOAD_CV_TEST_DAYS,
    GEN_LOAD_HISTORICAL_FOLDS,
    GEN_LOAD_HOLDOUT_DAYS,
    GEN_LOAD_MAX_TRAIN_HOURS,
    GEN_LOAD_TARGETS,
    HOLDOUT_DAYS,
)
from energy_forecasting.features.forecast_coverage import (
    assert_price_holdout_forecast_coverage,
    monthly_artifact_coverage,
)
from energy_forecasting.modeling.cv import TimeSeriesSplitter, carve_holdout
from energy_forecasting.modeling.datasets import DATASET_DIR
from energy_forecasting.modeling.gen_load import _load_tso_data, _load_weather_data

MODEL_TYPES = ("lgbmregressor", "xgbregressor", "elasticnet")


def _json_default(value: Any) -> str:
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _first_existing_dataset(target: str, region: str) -> Path | None:
    for model_type in MODEL_TYPES:
        path = DATASET_DIR / f"{target}_{region}_{model_type}.parquet"
        if path.exists():
            return path
    return None


def _input_overlap_index(target: str, region: str) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    tso = _load_tso_data(region)
    weather_history = _load_weather_data(target, region, source="history")
    weather_hist_forecast = _load_weather_data(target, region, source="hist_forecast")

    # Final training rows are built from actual weather. The parallel
    # hist_forecast matrix is only consumed for CV test folds and holdout rows,
    # so its start is a coverage gate for export rows, not the dataset start.
    start = max(tso.index.min(), weather_history.index.min())
    end = min(tso.index.max(), weather_history.index.max(), weather_hist_forecast.index.max())
    return pd.date_range(start=start, end=end, freq="h"), weather_hist_forecast.index.min()


def _span_source_index(
    target: str,
    region: str,
    source: str,
) -> tuple[pd.DatetimeIndex, str, pd.Timestamp | None]:
    if source == "inputs":
        index, hist_forecast_start = _input_overlap_index(target, region)
        return index, "current_inputs", hist_forecast_start
    path = _first_existing_dataset(target, region)
    if path is None:
        return pd.DatetimeIndex([]), "missing_dataset", None
    df = pd.read_parquet(path)
    return pd.DatetimeIndex(df.index), str(path), None


def _expected_export_span(
    index: pd.DatetimeIndex,
    hist_forecast_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(index).sort_values()
    if len(index) > GEN_LOAD_MAX_TRAIN_HOURS:
        index = index[-GEN_LOAD_MAX_TRAIN_HOURS:]

    pool_idx, holdout_idx = carve_holdout(index, GEN_LOAD_HOLDOUT_DAYS)
    pool_index = index[pool_idx]
    holdout_index = index[holdout_idx]

    cv = TimeSeriesSplitter(
        n_splits=GEN_LOAD_HISTORICAL_FOLDS,
        test_days=GEN_LOAD_CV_TEST_DAYS,
        mode="sliding",
    )
    oof_parts = [pool_index[test_idx] for _train_idx, test_idx in cv.split(pool_index)]
    if oof_parts:
        oof_index = oof_parts[0]
        for part in oof_parts[1:]:
            oof_index = oof_index.union(part)
    else:
        oof_index = pd.DatetimeIndex([])

    export_index = oof_index.union(holdout_index)
    export_start = export_index.min() if len(export_index) else None
    hist_ok = None
    if hist_forecast_start is not None and export_start is not None:
        hist_ok = export_start >= hist_forecast_start
    return {
        "dataset_start": index.min().isoformat(),
        "dataset_end": index.max().isoformat(),
        "oof_start": oof_index.min().isoformat() if len(oof_index) else None,
        "oof_end": oof_index.max().isoformat() if len(oof_index) else None,
        "holdout_start": holdout_index.min().isoformat(),
        "holdout_end": holdout_index.max().isoformat(),
        "export_start": export_start.isoformat() if export_start is not None else None,
        "export_end": export_index.max().isoformat() if len(export_index) else None,
        "export_rows": int(len(export_index)),
        "folds_configured": GEN_LOAD_HISTORICAL_FOLDS,
        "folds_actual": len(oof_parts),
        "max_train_hours": GEN_LOAD_MAX_TRAIN_HOURS,
        "hist_forecast_start": hist_forecast_start.isoformat() if hist_forecast_start is not None else None,
        "export_start_has_hist_forecast_coverage": hist_ok,
    }


def cmd_spans(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for target, cfg in GEN_LOAD_TARGETS.items():
        for region in cfg["regions"]:
            index, source_label, hist_forecast_start = _span_source_index(
                target, region, args.source
            )
            if index.empty:
                rows.append({"target": target, "region": region, "status": source_label})
                continue
            span = _expected_export_span(index, hist_forecast_start=hist_forecast_start)
            rows.append(
                {
                    "target": target,
                    "region": region,
                    "status": "ok",
                    "source": source_label,
                    **span,
                }
            )
    print(json.dumps(rows, indent=2, default=_json_default))


def cmd_coverage(args: argparse.Namespace) -> None:
    merged_path = Path(args.merged_path)
    raw = pd.read_parquet(merged_path)
    monthly = monthly_artifact_coverage(raw, start=args.start, end=args.end)
    monthly_records = {
        str(period): {
            "own": int(row["own"]),
            "smard": int(row["smard"]),
            "actual": int(row["actual"]),
            "missing": int(row["missing"]),
            "total": int(row["total"]),
            "own_fraction": float(row["own_fraction"]),
        }
        for period, row in monthly.iterrows()
    }

    report: dict[str, Any] = {"monthly": monthly_records}
    price_max = DATASET_DIR / "price_max.parquet"
    if price_max.exists():
        dataset_index = pd.DatetimeIndex(pd.read_parquet(price_max).index)
        report["price_max_holdout"] = assert_price_holdout_forecast_coverage(
            dataset_index,
            raw,
            holdout_days=HOLDOUT_DAYS,
            min_own_fraction=0.0,
            context="price_max audit",
        )
    print(json.dumps(report, indent=2, default=_json_default))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    spans = sub.add_parser("spans", help="Dry-calculate expected gen/load export spans")
    spans.add_argument(
        "--source",
        choices=["inputs", "datasets"],
        default="inputs",
        help="Use current TSO/weather input overlap (default) or stale generated datasets.",
    )
    spans.set_defaults(func=cmd_spans)

    coverage = sub.add_parser("coverage", help="Report current artifact coverage")
    coverage.add_argument(
        "--merged-path",
        default=str(PROCESSED_DATA_DIR / "merged.parquet"),
        help="Merged parquet path",
    )
    coverage.add_argument("--start", default="2022-01-15", help="Coverage start timestamp")
    coverage.add_argument("--end", default=None, help="Coverage end timestamp")
    coverage.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
