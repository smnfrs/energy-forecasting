"""Coverage checks for source-neutral forecast inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecasting.config import PROCESSED_DATA_DIR
from energy_forecasting.config.modeling import (
    FORECAST_ARTIFACT_MIN_MONTHLY_OWN_FRACTION,
    HOLDOUT_DAYS,
    PRICE_HOLDOUT_MIN_OWN_FORECAST_FRACTION,
)
from energy_forecasting.features.forecast_inputs import (
    forecast_source_counts,
    forecast_source_labels,
)
from energy_forecasting.modeling.cv import carve_holdout

SOURCE_ORDER = ("own", "smard", "actual", "missing")


def _coverage_summary(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.get(k, 0) for k in SOURCE_ORDER)
    own_fraction = counts.get("own", 0) / total if total else 0.0
    return {"counts": {k: int(counts.get(k, 0)) for k in SOURCE_ORDER}, "total": total, "own_fraction": own_fraction}


def assert_price_holdout_forecast_coverage(
    dataset_index: pd.DatetimeIndex,
    raw_merged: pd.DataFrame,
    *,
    holdout_days: int = HOLDOUT_DAYS,
    min_own_fraction: float = PRICE_HOLDOUT_MIN_OWN_FORECAST_FRACTION,
    forecast_root: Path | None = None,
    context: str = "price holdout",
) -> dict[str, Any]:
    """Require own forecast-artifact coverage over the exact price holdout split."""
    dataset_index = pd.DatetimeIndex(dataset_index)
    if dataset_index.empty:
        raise RuntimeError(f"{context} forecast coverage failed: empty dataset index")

    _pool_idx, holdout_idx = carve_holdout(dataset_index, holdout_days)
    holdout_index = dataset_index[holdout_idx]
    labels = forecast_source_labels(raw_merged, forecast_root=forecast_root)
    counts = forecast_source_counts(labels, holdout_index)
    summary = _coverage_summary(counts)
    summary.update(
        {
            "holdout_start": holdout_index.min().isoformat(),
            "holdout_end": holdout_index.max().isoformat(),
            "min_own_fraction": min_own_fraction,
        }
    )

    failures: list[str] = []
    if summary["own_fraction"] < min_own_fraction:
        failures.append(
            f"own_fraction={summary['own_fraction']:.2%} < {min_own_fraction:.2%}"
        )
    if counts.get("actual", 0):
        failures.append(f"actual fallback rows={counts['actual']}")
    if counts.get("missing", 0):
        failures.append(f"missing rows={counts['missing']}")

    if failures:
        raise RuntimeError(
            f"{context} forecast coverage failed over "
            f"{summary['holdout_start']} -> {summary['holdout_end']}: "
            f"{'; '.join(failures)}; counts={summary['counts']}"
        )
    return summary


def monthly_artifact_coverage(
    raw_merged: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2022-01-15",
    end: str | pd.Timestamp | None = None,
    forecast_root: Path | None = None,
) -> pd.DataFrame:
    """Return monthly all-five-own-artifact coverage for merged-data rows."""
    labels = forecast_source_labels(raw_merged, forecast_root=forecast_root)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else labels.index.max()
    window = labels.loc[(labels.index >= start_ts) & (labels.index <= end_ts)]
    if window.empty:
        return pd.DataFrame(columns=[*SOURCE_ORDER, "total", "own_fraction"])

    monthly = pd.crosstab(window.index.to_period("M"), window)
    monthly = monthly.reindex(columns=SOURCE_ORDER, fill_value=0).astype(int)
    monthly["total"] = monthly[list(SOURCE_ORDER)].sum(axis=1)
    monthly["own_fraction"] = monthly["own"] / monthly["total"].where(monthly["total"] > 0, pd.NA)
    return monthly


def assert_monthly_artifact_coverage(
    *,
    merged_path: Path | None = None,
    start: str | pd.Timestamp = "2022-01-15",
    end: str | pd.Timestamp | None = None,
    min_own_fraction: float = FORECAST_ARTIFACT_MIN_MONTHLY_OWN_FRACTION,
    forecast_root: Path | None = None,
) -> dict[str, Any]:
    """Fail if any monitored month has insufficient all-five own coverage."""
    merged_path = merged_path or (PROCESSED_DATA_DIR / "merged.parquet")
    raw = pd.read_parquet(merged_path)
    monthly = monthly_artifact_coverage(raw, start=start, end=end, forecast_root=forecast_root)
    if monthly.empty:
        raise RuntimeError(f"Forecast artifact coverage failed: no rows from {start} to {end}")

    bad = monthly[monthly["own_fraction"] < min_own_fraction]
    if not bad.empty:
        bad_summary = {
            str(period): {
                "own_fraction": float(row["own_fraction"]),
                "own": int(row["own"]),
                "smard": int(row["smard"]),
                "actual": int(row["actual"]),
                "missing": int(row["missing"]),
                "total": int(row["total"]),
            }
            for period, row in bad.iterrows()
        }
        raise RuntimeError(
            "Forecast artifact monthly coverage failed: "
            f"{bad_summary}; required own_fraction >= {min_own_fraction:.2%}"
        )

    return {
        "start": str(start),
        "end": str(end) if end is not None else raw.index.max().isoformat(),
        "min_own_fraction": min_own_fraction,
        "months_checked": int(len(monthly)),
        "min_observed_own_fraction": float(monthly["own_fraction"].min()),
    }
