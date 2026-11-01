"""Tests for the CI own-forecast-history freshness check."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from energy_forecasting.deploy.history_freshness import (
    StaleHistoryError,
    check_history,
    main,
)
from energy_forecasting.features.forecast_inputs import _ARTIFACT_FILES

FILES = list(_ARTIFACT_FILES.values())


def _merged(actuals_end: str = "2026-10-30 23:00", forward_hours: int = 48) -> pd.DataFrame:
    """Merged frame: actuals up to ``actuals_end``, then forecast-only rows."""
    idx = pd.date_range("2026-09-01", pd.Timestamp(actuals_end) + pd.Timedelta(hours=forward_hours), freq="h")
    df = pd.DataFrame(
        {
            "stromverbrauch_gesamt_(netzlast)": 50.0,
            "stromerzeugung_wind_onshore": 10.0,
            "stromerzeugung_wind_offshore": 5.0,
            "stromerzeugung_photovoltaik": 8.0,
            "stromerzeugung_gesamt": 53.0,
        },
        index=idx,
    )
    df.loc[df.index > pd.Timestamp(actuals_end), df.columns] = np.nan
    return df


def _write_history(root: Path, end: str, *, skip: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2026-06-01", end, freq="h", tz="UTC")
    for name in FILES:
        if name == skip:
            continue
        pd.DataFrame({"y_pred": 1.0, "y_true": 1.0}, index=idx).to_parquet(root / name)


def _tarball(path: Path, src_root: Path, *, extra: dict[str, bytes] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name in FILES:
            f = src_root / name
            if f.exists():
                tf.add(f, arcname=f"data/processed/historical_forecasts/{name}")
        for arc, data in (extra or {}).items():
            info = tarfile.TarInfo(arc)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_fresh_warm_cache_is_ok(tmp_path):
    root = tmp_path / "hf"
    _write_history(root, "2026-11-05 23:00")  # warm cache holds forward rows
    result = check_history(_merged(), root)
    assert result.status == "ok"


def test_healed_history_slightly_behind_actuals_is_ok(tmp_path):
    root = tmp_path / "hf"
    _write_history(root, "2026-10-29 23:00")  # 24h behind actuals frontier
    assert check_history(_merged(), root).status == "ok"


def test_stale_cache_falls_back_to_fresh_tarball(tmp_path):
    root = tmp_path / "hf"
    _write_history(root, "2026-08-30 23:00")
    fresh = tmp_path / "fresh"
    _write_history(fresh, "2026-10-30 23:00")
    tar = _tarball(tmp_path / "hf.tar.gz", fresh)

    result = check_history(_merged(), root, release_tarball=tar)

    assert result.status == "fallback"
    replaced = pd.read_parquet(root / FILES[0])
    assert replaced.index.max() == pd.Timestamp("2026-10-30 23:00", tz="UTC")


def test_stale_cache_and_stale_tarball_fails_with_clear_message(tmp_path):
    root = tmp_path / "hf"
    _write_history(root, "2026-08-30 23:00")
    old = tmp_path / "old"
    _write_history(old, "2026-07-28 23:00")
    tar = _tarball(tmp_path / "hf.tar.gz", old)

    with pytest.raises(StaleHistoryError, match="stale.*Release tarball is also stale"):
        check_history(_merged(), root, release_tarball=tar)
    # The stale cache must be left untouched on failure.
    assert pd.read_parquet(root / FILES[0]).index.max() == pd.Timestamp("2026-08-30 23:00", tz="UTC")


def test_missing_history_without_tarball_fails(tmp_path):
    with pytest.raises(StaleHistoryError, match="no Release tarball available"):
        check_history(_merged(), tmp_path / "empty")


def test_one_lagging_artifact_makes_history_stale(tmp_path):
    # "Own" needs all five files, so a missing/lagging one defeats freshness.
    root = tmp_path / "hf"
    _write_history(root, "2026-10-30 23:00", skip=FILES[0])
    with pytest.raises(StaleHistoryError):
        check_history(_merged(), root)


def test_tarball_extraction_ignores_unsafe_and_unrelated_members(tmp_path):
    root = tmp_path / "hf"
    _write_history(root, "2026-08-30 23:00")
    fresh = tmp_path / "fresh"
    _write_history(fresh, "2026-10-30 23:00")
    tar = _tarball(
        tmp_path / "hf.tar.gz",
        fresh,
        extra={"../../escape.parquet": b"x", "data/other.txt": b"y"},
    )
    check_history(_merged(), root, release_tarball=tar)
    assert not (tmp_path.parent / "escape.parquet").exists()
    assert sorted(p.name for p in root.iterdir()) == sorted(FILES)


def test_cli_returns_nonzero_on_stale_history(tmp_path, capsys):
    merged_path = tmp_path / "merged.parquet"
    _merged().to_parquet(merged_path)
    root = tmp_path / "hf"
    _write_history(root, "2026-08-30 23:00")

    code = main(["--merged", str(merged_path), "--root", str(root)])

    assert code == 1
    assert "Own-forecast history is stale" in capsys.readouterr().err


def test_cli_returns_zero_when_fresh(tmp_path, capsys):
    merged_path = tmp_path / "merged.parquet"
    _merged().to_parquet(merged_path)
    root = tmp_path / "hf"
    _write_history(root, "2026-10-30 23:00")

    assert main(["--merged", str(merged_path), "--root", str(root)]) == 0
    assert "fresh" in capsys.readouterr().out
