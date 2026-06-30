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
