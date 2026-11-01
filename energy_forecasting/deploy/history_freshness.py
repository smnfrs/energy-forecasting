"""Freshness check for the restored own-forecast history (used by CI).

The daily workflow restores ``data/processed/historical_forecasts/`` from an
Actions cache, and falls back to the ``gen_load_hf.tar.gz`` Release asset on a
cold cache. Either source can be stale: caches are evicted after 7 days unused
and the Release asset is only refreshed by a manual heal. A stale restore
otherwise surfaces much later, as a confusing monthly-coverage failure inside
inference.

This module measures how far the restored history lags the actuals frontier of
``merged.parquet``, using the same definitions as the coverage gate
(``forecast_source_labels``): a row is "own" only when all five national
artifacts have a ``y_pred``. If the lag is too large it tries the Release
tarball, and fails fast with a clear message if neither source is fresh enough.

The coverage gate stays authoritative; the default threshold is deliberately
loose so this only catches gross staleness and never blocks a healthy run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from energy_forecasting.config import HISTORICAL_FORECASTS_DIR, PROCESSED_DATA_DIR
from energy_forecasting.features.forecast_inputs import (
    _ARTIFACT_FILES,
    _derive_from_actuals,
    _load_artifact_layer,
)

DEFAULT_MAX_GAP_HOURS = 72


class StaleHistoryError(RuntimeError):
    """Neither the restored history nor the Release fallback is fresh enough."""


@dataclass(frozen=True)
class FreshnessResult:
    status: str  # "ok" | "fallback"
    message: str


def own_frontier(merged_index: pd.DatetimeIndex, root: Path) -> pd.Timestamp | None:
    """Last merged row for which all five artifacts have a forecast, or None."""
    layer = _load_artifact_layer(merged_index, root=root)
    own = layer.notna().all(axis=1)
    return own[own].index.max() if own.any() else None


def actual_frontier(merged: pd.DataFrame) -> pd.Timestamp:
    """Last merged row with complete actuals (what the gate scores against)."""
    actual = _derive_from_actuals(merged).notna().all(axis=1)
    if not actual.any():
        raise StaleHistoryError("merged data has no rows with complete actuals")
    return actual[actual].index.max()


def _gap_hours(actual: pd.Timestamp, own: pd.Timestamp | None) -> float:
    """Hours the own history lags the actuals frontier (<= 0 means up to date)."""
    if own is None:
        return float("inf")
    return (actual - own) / pd.Timedelta(hours=1)


def _lag(gap: float) -> str:
    return "no history found" if gap == float("inf") else f"lag {max(gap, 0):.0f}h"


def _extract_artifacts(tarball: Path, dest: Path) -> None:
    """Extract only the five artifact parquet files, flat, ignoring any paths."""
    wanted = set(_ARTIFACT_FILES.values())
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(dest / name, "wb") as out:
                shutil.copyfileobj(src, out)


def check_history(
    merged: pd.DataFrame,
    root: Path,
    *,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
    release_tarball: Path | None = None,
) -> FreshnessResult:
    """Return ok/fallback, or raise ``StaleHistoryError`` with a clear message."""
    actual = actual_frontier(merged)
    current = own_frontier(merged.index, root)
    gap = _gap_hours(actual, current)
    if gap <= max_gap_hours:
        return FreshnessResult(
            "ok",
            f"Restored history is fresh: own frontier {current}, actuals frontier "
            f"{actual} ({_lag(gap)} <= {max_gap_hours:.0f}h)",
        )

    fallback_note = "no Release tarball available"
    if release_tarball is not None and Path(release_tarball).exists():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _extract_artifacts(Path(release_tarball), tmp_dir)
            candidate = own_frontier(merged.index, tmp_dir)
            cand_gap = _gap_hours(actual, candidate)
            if cand_gap <= max_gap_hours:
                root.mkdir(parents=True, exist_ok=True)
                for name in _ARTIFACT_FILES.values():
                    shutil.copyfile(tmp_dir / name, root / name)
                return FreshnessResult(
                    "fallback",
                    f"Restored history was stale (own frontier {current}, {_lag(gap)}); "
                    f"replaced it from the Release tarball "
                    f"(own frontier {candidate}, {_lag(cand_gap)})",
                )
            fallback_note = f"Release tarball is also stale (own frontier {candidate})"

    raise StaleHistoryError(
        f"Own-forecast history is stale: own frontier {current}, actuals frontier "
        f"{actual} ({_lag(gap)}, limit {max_gap_hours:.0f}h); {fallback_note}. "
        "Run the gen/load heal (`train gen-load --reuse-params`), upload a fresh "
        "gen_load_hf.tar.gz to the v1.0.0 Release, and delete the stale "
        "deploy-state cache."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--merged", type=Path, default=PROCESSED_DATA_DIR / "merged.parquet")
    parser.add_argument("--root", type=Path, default=HISTORICAL_FORECASTS_DIR)
    parser.add_argument("--release-tarball", type=Path, default=None)
    parser.add_argument("--max-gap-hours", type=float, default=DEFAULT_MAX_GAP_HOURS)
    args = parser.parse_args(argv)

    merged = pd.read_parquet(args.merged)
    try:
        result = check_history(
            merged,
            args.root,
            max_gap_hours=args.max_gap_hours,
            release_tarball=args.release_tarball,
        )
    except StaleHistoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(result.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
