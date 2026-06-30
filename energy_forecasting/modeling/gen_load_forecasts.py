"""Loader for gen/load forecast features consumed by price models.

Reads leak-free per-target predictions written by Stage 5b training
(``data/processed/historical_forecasts/{target}_{region}.parquet``) and
exposes them as ``_derived_forecast_*`` columns aligned to a requested
timestamp index.

At training time, training-period rows are OOF predictions and
holdout-period rows are final-model predictions. At inference the same
files are populated with that day's live forecasts, so the price pipeline
reads from a single location with no run-id lookups.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.config import HISTORICAL_FORECASTS_DIR

# Map _derived_forecast_* column → (target file stem, region suffix).
# `forecast_residual` is derived from the other four (load minus generation).
FORECAST_TARGETS: dict[str, tuple[str, str]] = {
    "_derived_forecast_wind_on": ("wind_onshore", "DE_NATIONAL"),
    "_derived_forecast_wind_off": ("wind_offshore", "DE_NATIONAL"),
    "_derived_forecast_solar": ("solar", "DE_NATIONAL"),
    "_derived_forecast_load": ("load", "DE_NATIONAL"),
    "_derived_forecast_gen_load_diff": ("gen_load_diff", "DE_NATIONAL"),
}

# Inputs required to derive `forecast_residual`.
_RESIDUAL_INPUTS = (
    "_derived_forecast_load",
    "_derived_forecast_wind_on",
    "_derived_forecast_wind_off",
    "_derived_forecast_solar",
)


def _file_path(target: str, region: str, root: Path | None = None) -> Path:
    root = root if root is not None else HISTORICAL_FORECASTS_DIR
    return root / f"{target}_{region}.parquet"


def _align_tz(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Bring a source series's index tz into agreement with the target.

    SMARD/Open-Meteo merged data is stored tz-naive but conventionally
    represents UTC moments; the 5b artifacts are written tz-aware UTC.
    Reindexing across the mismatch silently returns NaN, so normalise
    before the lookup.
    """
    src_tz = series.index.tz
    dst_tz = target_index.tz
    if src_tz is None and dst_tz is None:
        return series
    if src_tz is not None and dst_tz is None:
        return series.tz_convert("UTC").tz_localize(None)
    if src_tz is None and dst_tz is not None:
        return series.tz_localize("UTC").tz_convert(dst_tz)
    return series.tz_convert(dst_tz)


def load_gen_load_forecasts(
    index: pd.DatetimeIndex,
    columns: list[str] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of ``_derived_forecast_*`` columns aligned to ``index``.

    Parameters
    ----------
    index
        Target DatetimeIndex (typically the merged dataset's index).
    columns
        Subset of ``_derived_forecast_*`` columns to materialise.
        Defaults to all six columns including ``_derived_forecast_residual``.
    root
        Override the source directory (used in tests).

    Raises
    ------
    FileNotFoundError
        If any required source parquet is missing. The error message lists
        every missing file so the gap is visible at a glance.
    """
    requested = list(columns) if columns is not None else _all_columns()
    needed_base = _resolve_base_columns(requested)

    root = root if root is not None else HISTORICAL_FORECASTS_DIR

    missing: list[Path] = []
    file_specs: dict[str, Path] = {}
    for col in needed_base:
        target, region = FORECAST_TARGETS[col]
        path = _file_path(target, region, root=root)
        file_specs[col] = path
        if not path.exists():
            missing.append(path)

    if missing:
        msg_lines = [
            "load_gen_load_forecasts: missing historical_forecasts parquet(s).",
            "Each file is produced by Stage 5b training; re-run gen/load "
            "training before price feature dataset preparation.",
            "Missing:",
        ]
        msg_lines.extend(f"  - {p}" for p in missing)
        raise FileNotFoundError("\n".join(msg_lines))

    out = pd.DataFrame(index=index)
    for col, path in file_specs.items():
        series = pd.read_parquet(path)["y_pred"]
        series = _align_tz(series, index)
        out[col] = series.reindex(index)

    if "_derived_forecast_residual" in requested:
        out["_derived_forecast_residual"] = (
            out["_derived_forecast_load"]
            - out["_derived_forecast_wind_on"]
            - out["_derived_forecast_wind_off"]
            - out["_derived_forecast_solar"]
        )

    return out[requested]


def _all_columns() -> list[str]:
    return [*FORECAST_TARGETS.keys(), "_derived_forecast_residual"]


def _resolve_base_columns(requested: list[str]) -> list[str]:
    """Expand requests so derived columns (residual) pull in their inputs."""
    base: list[str] = []
    for col in requested:
        if col == "_derived_forecast_residual":
            for inp in _RESIDUAL_INPUTS:
                if inp not in base:
                    base.append(inp)
        elif col in FORECAST_TARGETS:
            if col not in base:
                base.append(col)
        else:
            raise ValueError(
                f"Unknown gen/load forecast column {col!r}. "
                f"Expected one of {_all_columns()}."
            )
    return base
