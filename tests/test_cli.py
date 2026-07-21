"""Smoke tests for CLI commands — verify they exist and --help works."""

from energy_forecasting.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.output
    assert "update" in result.output


def test_download_help():
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "smard" in result.output
    assert "weather" in result.output
    assert "commodities" in result.output


def test_update_help():
    result = runner.invoke(app, ["update", "--help"])
    assert result.exit_code == 0
    assert "all" in result.output
    assert "smard" in result.output


def test_download_smard_help():
    result = runner.invoke(app, ["download", "smard", "--help"])
    assert result.exit_code == 0
    assert "region" in result.output.lower()


def test_download_weather_help():
    result = runner.invoke(app, ["download", "weather", "--help"])
    assert result.exit_code == 0
    assert "all" in result.output.lower()


def test_process_help():
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "output" in result.output.lower()


def test_commodity_commands_use_sequential_execution():
    """yfinance is not thread-safe. Commodity commands must not use _run_parallel.

    This test inspects the source code of the CLI functions to ensure
    commodity downloads/updates use _run_sequential, not _run_parallel.
    Catches regressions from someone 'optimizing' to parallel execution.
    """
    import inspect

    from energy_forecasting.cli import (
        download_all_sources,
        download_commodities,
        update_commodities,
    )

    for fn in [download_commodities, update_commodities]:
        source = inspect.getsource(fn)
        assert "_run_sequential" in source, (
            f"{fn.__name__} must use _run_sequential (yfinance is not thread-safe)"
        )
        assert "_run_parallel" not in source, (
            f"{fn.__name__} must not use _run_parallel (yfinance is not thread-safe)"
        )

    # download_all_sources uses inner functions; check the full source
    source = inspect.getsource(download_all_sources)
    assert "_download_other" in source
    # The _download_other inner function must use _run_sequential
    # Find the _download_other block and verify
    other_idx = source.index("_download_other")
    other_block = source[other_idx : other_idx + 300]
    assert "_run_sequential" in other_block, (
        "download_all _download_other must use _run_sequential (yfinance is not thread-safe)"
    )


def test_batch_helpers_return_failures():
    from energy_forecasting.cli import _run_parallel, _run_sequential

    def ok():
        return None

    def bad():
        raise RuntimeError("boom")

    assert _run_sequential([(ok, "ok"), (bad, "bad")], label="seq") == ["bad"]
    assert _run_parallel([(ok, "ok"), (bad, "bad")], max_workers=2, label="par") == ["bad"]


def test_exit_if_failures_exits_nonzero():
    import typer
    from energy_forecasting.cli import _exit_if_failures

    try:
        _exit_if_failures(["bad"])
    except typer.Exit as exc:
        assert exc.exit_code == 1
    else:
        raise AssertionError("expected typer.Exit")


def _hf_frame(start, periods, val):
    import numpy as np
    import pandas as pd

    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "y_true": np.full(periods, val + 1.0),
            "y_pred": np.full(periods, float(val)),
            "y_lower": np.full(periods, np.nan),
            "y_upper": np.full(periods, np.nan),
        },
        index=idx,
    )


def test_export_historical_forecasts_merge_preserves_earlier_rows(tmp_path, monkeypatch):
    """The OOF window slides forward on regen; a fresh export must union with
    the existing file so earlier already-generated forecasts survive instead of
    being dropped back to a SMARD fallback."""
    import energy_forecasting.cli as cli
    import energy_forecasting.config as config
    import energy_forecasting.modeling.mlflow_utils as mlflow_utils
    import mlflow

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(mlflow_utils, "ensure_mlflow_tracking", lambda: None)

    windows = {}  # run_id -> oof parquet path

    class FakeClient:
        def download_artifacts(self, run_id, artifact_path):
            if artifact_path.endswith("oof_predictions.parquet"):
                return str(windows[run_id])
            raise FileNotFoundError(artifact_path)  # no holdout artifact

    monkeypatch.setattr(mlflow, "MlflowClient", lambda *a, **k: FakeClient())

    # First export: an earlier window (Jan 01 → Jan 03).
    w1 = tmp_path / "w1.parquet"
    _hf_frame("2025-01-01", 48, 10.0).to_parquet(w1)
    windows["run1"] = w1
    cli._export_historical_forecasts("load", "DE_50HZ", "run1")

    # Second export: a *later* window (Jan 02 → Jan 05) — Jan 01 now out of window.
    w2 = tmp_path / "w2.parquet"
    _hf_frame("2025-01-02", 72, 20.0).to_parquet(w2)
    windows["run2"] = w2
    cli._export_historical_forecasts("load", "DE_50HZ", "run2")

    import pandas as pd

    out = pd.read_parquet(tmp_path / "historical_forecasts" / "load_DE_50HZ.parquet")
    # Jan-01 (only in the first window) must be preserved...
    assert out.index.min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert out.index.max() == pd.Timestamp("2025-01-04 23:00", tz="UTC")
    assert out.loc[pd.Timestamp("2025-01-01", tz="UTC"), "y_pred"] == 10.0
    # ...and the fresh window wins on overlap.
    assert out.loc[pd.Timestamp("2025-01-03", tz="UTC"), "y_pred"] == 20.0


def test_aggregate_national_merge_preserves_earlier_rows(tmp_path, monkeypatch):
    """National aggregate must keep earlier national rows outside the current
    per-region window rather than overwriting with only the fresh sum."""
    import energy_forecasting.cli as cli
    import energy_forecasting.config as config
    import pandas as pd

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)
    hf = tmp_path / "historical_forecasts"
    hf.mkdir()

    # Two regions cover Jan 03 → Jan 05 only.
    _hf_frame("2025-01-03", 72, 4.0).to_parquet(hf / "wind_onshore_DE_50HZ.parquet")
    _hf_frame("2025-01-03", 72, 6.0).to_parquet(hf / "wind_onshore_DE_TENNET.parquet")
    # An existing national file already holds earlier rows (Jan 01 → Jan 05).
    _hf_frame("2025-01-01", 120, 99.0).to_parquet(hf / "wind_onshore_DE_NATIONAL.parquet")

    cli._aggregate_national_historical_forecasts(
        [("wind_onshore", "DE_50HZ"), ("wind_onshore", "DE_TENNET")]
    )

    out = pd.read_parquet(hf / "wind_onshore_DE_NATIONAL.parquet")
    # Earlier national rows outside the regional window are preserved...
    assert out.index.min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert out.loc[pd.Timestamp("2025-01-01", tz="UTC"), "y_pred"] == 99.0
    # ...and the overlap is the fresh sum of regions (4 + 6 = 10).
    assert out.loc[pd.Timestamp("2025-01-03", tz="UTC"), "y_pred"] == 10.0
