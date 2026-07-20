"""Source-neutral generation/load forecast inputs for price models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_forecasting.config import HISTORICAL_FORECASTS_DIR
from energy_forecasting.modeling.gen_load_forecasts import _align_tz

FORECAST_COLUMNS: tuple[str, ...] = (
    "forecast_load",
    "forecast_gen_wind_on",
    "forecast_gen_wind_off",
    "forecast_gen_solar",
    "forecast_gen_wind_pv",
    "forecast_gen_total",
    "forecast_gen_other",
    "forecast_residual_load",
)

_ARTIFACT_FILES: dict[str, str] = {
    "forecast_gen_wind_on": "wind_onshore_DE_NATIONAL.parquet",
    "forecast_gen_wind_off": "wind_offshore_DE_NATIONAL.parquet",
    "forecast_gen_solar": "solar_DE_NATIONAL.parquet",
    "forecast_load": "load_DE_NATIONAL.parquet",
    "gen_load_diff_forecast": "gen_load_diff_DE_NATIONAL.parquet",
}

_SMARD_COLUMNS: dict[str, str] = {
    "forecast_load": "prognostizierter_verbrauch_gesamt",
    "forecast_gen_wind_on": "prognostizierte_erzeugung_onshore",
    "forecast_gen_wind_off": "prognostizierte_erzeugung_offshore",
    "forecast_gen_solar": "prognostizierte_erzeugung_photovoltaik",
    "forecast_gen_wind_pv": "prognostizierte_erzeugung_wind_und_photovoltaik",
    "forecast_gen_total": "prognostizierte_erzeugung_gesamt",
    "forecast_gen_other": "prognostizierte_erzeugung_sonstige",
    "forecast_residual_load": "prognostizierter_verbrauch_residuallast",
}

_ACTUAL_COLUMNS: dict[str, str] = {
    "forecast_load": "stromverbrauch_gesamt_(netzlast)",
    "forecast_gen_wind_on": "stromerzeugung_wind_onshore",
    "forecast_gen_wind_off": "stromerzeugung_wind_offshore",
    "forecast_gen_solar": "stromerzeugung_photovoltaik",
}

_ACTUAL_TOTAL_CANDIDATES: tuple[str, ...] = (
    "stromerzeugung_gesamt",
    "total_generation",
    "_derived_total_generation",
)


def build_forecast_columns(
    df: pd.DataFrame,
    *,
    strict_index: pd.DatetimeIndex | None = None,
    forecast_root: Path | None = None,
) -> pd.DataFrame:
    """Append source-neutral ``forecast_*`` columns for price features.

    Historical rows use a coherent waterfall:

    ``own gen/load forecast artifacts -> SMARD forecasts -> actuals``.

    Strict rows, used for live D+1 price inference, must be fully covered by
    the five own gen/load forecast artifacts. No SMARD or actual fallback is
    allowed for those rows.
    """
    out = df.copy()
    artifact = _load_artifact_layer(out.index, root=forecast_root)
    own_layer = _derive_from_artifacts(artifact)
    smard_layer = _derive_from_smard(out)
    actual_layer = _derive_from_actuals(out)

    own_mask = artifact.notna().all(axis=1)
    smard_mask = smard_layer.notna().all(axis=1)
    actual_mask = actual_layer.notna().all(axis=1)

    combined = pd.DataFrame(index=out.index, columns=FORECAST_COLUMNS, dtype=float)
    combined.loc[actual_mask, FORECAST_COLUMNS] = actual_layer.loc[actual_mask, FORECAST_COLUMNS]
    combined.loc[smard_mask, FORECAST_COLUMNS] = smard_layer.loc[smard_mask, FORECAST_COLUMNS]
    combined.loc[own_mask, FORECAST_COLUMNS] = own_layer.loc[own_mask, FORECAST_COLUMNS]

    if strict_index is not None:
        _validate_strict_index(strict_index, artifact)
        combined.loc[strict_index, FORECAST_COLUMNS] = own_layer.loc[strict_index, FORECAST_COLUMNS]

    for col in FORECAST_COLUMNS:
        out[col] = combined[col]

    out.attrs["forecast_source_counts"] = {
        "own": int(own_mask.sum()),
        "smard": int((~own_mask & smard_mask).sum()),
        "actual": int((~own_mask & ~smard_mask & actual_mask).sum()),
        "missing": int((~own_mask & ~smard_mask & ~actual_mask).sum()),
    }
    return out



def forecast_source_labels(
    df: pd.DataFrame,
    *,
    forecast_root: Path | None = None,
) -> pd.Series:
    """Label each row by the coherent forecast layer used by the waterfall.

    Labels are one of ``own``, ``smard``, ``actual``, or ``missing``. They use
    the same all-or-none own-artifact rule as ``build_forecast_columns``: if any
    of the five primary artifacts is missing for a row, that row is not labelled
    ``own``.
    """
    artifact = _load_artifact_layer(df.index, root=forecast_root)
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


def forecast_source_counts(
    labels: pd.Series,
    index: pd.DatetimeIndex | None = None,
) -> dict[str, int]:
    """Return stable source-label counts for an optional index subset."""
    subset = labels if index is None else labels.reindex(index)
    counts = subset.value_counts(dropna=True)
    return {
        "own": int(counts.get("own", 0)),
        "smard": int(counts.get("smard", 0)),
        "actual": int(counts.get("actual", 0)),
        "missing": int(counts.get("missing", 0) + subset.isna().sum()),
    }

def _load_artifact_layer(index: pd.DatetimeIndex, root: Path | None = None) -> pd.DataFrame:
    root = root if root is not None else HISTORICAL_FORECASTS_DIR
    data = pd.DataFrame(index=index)
    for col, filename in _ARTIFACT_FILES.items():
        path = root / filename
        if not path.exists():
            data[col] = float("nan")
            continue
        series = pd.read_parquet(path)["y_pred"]
        data[col] = _normalize_artifact_series(series, index).reindex(index)
    return data.astype(float)


def _normalize_artifact_series(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    if series.index.tz is not None and target_index.tz is None:
        from energy_forecasting.data.merge import normalize_dst

        frame = normalize_dst(series.to_frame("__value__"))
        return frame["__value__"]

    aligned = _align_tz(series, target_index)
    if aligned.index.has_duplicates:
        aligned = aligned.groupby(level=0).mean()
    return _normalize_local_delivery_grid(aligned)


def _normalize_local_delivery_grid(series: pd.Series) -> pd.Series:
    """Normalize an already-local tz-naive delivery grid.

    ``normalize_dst`` is used for the production UTC artifact path. This
    defensive path handles test/manual artifacts that are already stored as
    local wall-clock labels, where ``normalize_dst`` cannot safely infer an
    original timezone conversion.
    """
    if series.index.tz is not None:
        return series

    values = series.copy()
    inserts: list[pd.Series] = []
    for day, day_series in values.groupby(values.index.normalize()):
        deduped = day_series.groupby(level=0).mean() if day_series.index.has_duplicates else day_series
        if len(deduped) != 23:
            continue
        missing_hours = sorted(set(range(24)) - set(deduped.index.hour))
        if not missing_hours:
            continue
        hour = missing_hours[0]
        ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        prev_ts = ts - pd.Timedelta(hours=1)
        next_ts = ts + pd.Timedelta(hours=1)
        if prev_ts in deduped.index and next_ts in deduped.index:
            inserts.append(pd.Series({ts: (deduped.loc[prev_ts] + deduped.loc[next_ts]) / 2}))

    if inserts:
        values = pd.concat([values, *inserts])
    if values.index.has_duplicates:
        values = values.groupby(level=0).mean()
    return values.sort_index()


def _derive_from_artifacts(artifact: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=artifact.index)
    for col in (
        "forecast_load",
        "forecast_gen_wind_on",
        "forecast_gen_wind_off",
        "forecast_gen_solar",
    ):
        out[col] = artifact[col]
    out["forecast_gen_wind_pv"] = (
        out["forecast_gen_wind_on"] + out["forecast_gen_wind_off"] + out["forecast_gen_solar"]
    )
    out["forecast_gen_total"] = out["forecast_load"] + artifact["gen_load_diff_forecast"]
    out["forecast_gen_other"] = out["forecast_gen_total"] - out["forecast_gen_wind_pv"]
    out["forecast_residual_load"] = out["forecast_load"] - out["forecast_gen_wind_pv"]
    return out


def _derive_from_smard(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for forecast_col, raw_col in _SMARD_COLUMNS.items():
        out[forecast_col] = df[raw_col] if raw_col in df.columns else float("nan")
    return out.astype(float)


def _derive_from_actuals(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for forecast_col, raw_col in _ACTUAL_COLUMNS.items():
        out[forecast_col] = df[raw_col] if raw_col in df.columns else float("nan")

    out["forecast_gen_wind_pv"] = (
        out["forecast_gen_wind_on"] + out["forecast_gen_wind_off"] + out["forecast_gen_solar"]
    )
    out["forecast_gen_total"] = _actual_total_generation(df, out.index)
    out["forecast_gen_other"] = out["forecast_gen_total"] - out["forecast_gen_wind_pv"]
    out["forecast_residual_load"] = out["forecast_load"] - out["forecast_gen_wind_pv"]
    return out.astype(float)


def _actual_total_generation(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    for col in _ACTUAL_TOTAL_CANDIDATES:
        if col in df.columns:
            return df[col]

    generation_cols = [c for c in df.columns if c.startswith("stromerzeugung_")]
    if generation_cols:
        return df[generation_cols].sum(axis=1)
    return pd.Series(float("nan"), index=index)


def _validate_strict_index(
    strict_index: pd.DatetimeIndex,
    artifact: pd.DataFrame,
) -> None:
    missing_index = strict_index.difference(artifact.index)
    if len(missing_index):
        raise RuntimeError(
            "Forecast artifact strict coverage failed: strict_index contains "
            f"{len(missing_index)} timestamp(s) outside the working frame."
        )

    strict_artifact = artifact.reindex(strict_index)
    bad_by_col = {
        col: strict_artifact.index[strict_artifact[col].isna()].tolist()
        for col in _ARTIFACT_FILES
        if strict_artifact[col].isna().any()
    }
    if bad_by_col:
        summary = ", ".join(f"{col}={len(ts)} missing" for col, ts in bad_by_col.items())
        raise RuntimeError(
            "Forecast artifact strict coverage failed for live D+1 rows: "
            f"{summary}. SMARD/actual fallback is forbidden in strict mode."
        )
