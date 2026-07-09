"""CLI entry point.

Usage:
    energy-forecasting download smard --region DE-LU
    energy-forecasting download smard-tso --tso 50Hertz
    energy-forecasting download weather --type offshore --tso TenneT
    energy-forecasting download weather --all
    energy-forecasting download commodities
    energy-forecasting update all
    energy-forecasting update smard
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(help="Energy Forecasting CLI")


# ── Helpers ────────────────────────────────────────────────────────


def _run_parallel(tasks: list, max_workers: int, label: str) -> list[str]:
    """Run callables in parallel, logging and returning every failure."""
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for fn, name in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception:
                logger.exception(f"Failed: {name}")
                failures.append(name)
                continue
            if isinstance(result, list):
                failures.extend(f"{name}/{item}" for item in result)
    if failures:
        logger.error(f"{label}: {len(failures)} task(s) failed: {failures}")
    return failures


def _run_sequential(tasks: list, label: str) -> list[str]:
    """Run callables one at a time, logging and returning every failure.

    Aborts early on RateLimitExhausted (hourly/daily API limits)
    since all subsequent tasks would fail too.
    """
    from energy_forecasting.data.weather import RateLimitExhausted

    failures: list[str] = []
    for fn, name in tasks:
        try:
            fn()
        except RateLimitExhausted as exc:
            logger.warning(f"Rate limit exhausted on {name}, aborting remaining tasks: {exc}")
            failures.append(name)
            break
        except Exception:
            logger.exception(f"Failed: {name}")
            failures.append(name)
    if failures:
        logger.error(f"{label}: {len(failures)} task(s) failed: {failures}")
    return failures


def _exit_if_failures(failures: list[str]) -> None:
    if failures:
        raise typer.Exit(code=1)


# ── Download commands ───────────────────────────────────────────────

download_app = typer.Typer(help="Download data from scratch")
app.add_typer(download_app, name="download")


@download_app.command("smard")
def download_smard(
    region: str = typer.Option("DE-LU", help="National region (DE-LU or DE-AT-LU)"),
    resolution: str = typer.Option("hour", help="Data resolution"),
):
    """Download national SMARD data for a region."""
    from energy_forecasting.data.sources import SmardSource

    source = SmardSource(region, resolution)
    source.download()


@download_app.command("smard-tso")
def download_smard_tso(
    tso: str = typer.Option(..., help="TSO name (50Hertz, Amprion, TenneT, TransnetBW, Creos)"),
    resolution: str = typer.Option("hour", help="Data resolution"),
):
    """Download per-TSO SMARD generation/load data."""
    from energy_forecasting.data.sources import SmardSource

    source = SmardSource(tso, resolution)
    source.download()


@download_app.command("weather")
def download_weather(
    asset_type: str = typer.Option(None, help="offshore, onshore, solar, or cities"),
    tso: str = typer.Option(None, help="TSO name"),
    all_types: bool = typer.Option(False, "--all", help="Download all type x TSO combinations"),
):
    """Download Open-Meteo weather data."""
    from energy_forecasting.config.smard import TSO_REGIONS
    from energy_forecasting.data.weather import OpenMeteoSource

    if all_types:
        tasks = []
        for at in ["offshore", "onshore", "solar", "cities"]:
            for t in TSO_REGIONS:
                source = OpenMeteoSource(at, t)
                if source.locations:
                    tasks.append((source.download, f"weather/{at}/{t}"))
        _exit_if_failures(_run_sequential(tasks, label="weather"))
    else:
        if not asset_type or not tso:
            raise typer.BadParameter("Provide --asset-type and --tso, or use --all")
        OpenMeteoSource(asset_type, tso).download()


@download_app.command("commodities")
def download_commodities():
    """Download all commodity sources (ICAP, Yahoo, FRED)."""
    from energy_forecasting.data.commodities import all_commodity_sources

    # Sequential: yfinance is not thread-safe (concurrent calls corrupt results)
    sources = all_commodity_sources()
    tasks = [(s.download, type(s).__name__) for s in sources]
    _exit_if_failures(_run_sequential(tasks, label="commodities"))


@download_app.command("energy-charts")
def download_energy_charts():
    """Download Energy Charts day-ahead prices (fallback source)."""
    from energy_forecasting.data.sources import EnergyChartsSource

    EnergyChartsSource().download()


@download_app.command("all")
def download_all_sources():
    """Download everything from scratch. Takes a long time.

    Weather runs sequentially (Open-Meteo free tier rate limit) but
    concurrently with SMARD and commodities to maximize throughput.
    """
    from energy_forecasting.config.smard import NATIONAL_REGIONS, TSO_REGIONS
    from energy_forecasting.data.commodities import all_commodity_sources
    from energy_forecasting.data.sources import EnergyChartsSource, SmardSource
    from energy_forecasting.data.weather import OpenMeteoSource

    # Build task lists
    smard_tasks = []
    for region in NATIONAL_REGIONS:
        s = SmardSource(region)
        smard_tasks.append((s.download, f"SMARD/{region}"))
    for tso_name in TSO_REGIONS:
        s = SmardSource(tso_name)
        smard_tasks.append((s.download, f"SMARD/{tso_name}"))

    weather_tasks = []
    for at in ["offshore", "onshore", "solar", "cities"]:
        for tso_name in TSO_REGIONS:
            source = OpenMeteoSource(at, tso_name)
            if source.locations:
                weather_tasks.append((source.download, f"weather/{at}/{tso_name}"))

    commodity_sources = all_commodity_sources()
    other_tasks = [(s.download, type(s).__name__) for s in commodity_sources]
    other_tasks.append((EnergyChartsSource().download, "EnergyCharts"))

    # Run all three groups concurrently:
    #   - SMARD: 7 regions in parallel (each has internal 8-worker pool)
    #   - Weather: sequential (Open-Meteo rate limit)
    #   - Commodities + Energy Charts: all in parallel
    def _download_smard():
        logger.info(f"Downloading SMARD ({len(smard_tasks)} regions in parallel)")
        return _run_parallel(smard_tasks, max_workers=len(smard_tasks), label="SMARD")

    def _download_weather():
        logger.info(f"Downloading weather ({len(weather_tasks)} sources, sequential)")
        return _run_sequential(weather_tasks, label="weather")

    def _download_other():
        # Sequential: yfinance is not thread-safe (concurrent calls corrupt results)
        logger.info(
            f"Downloading commodities + Energy Charts ({len(other_tasks)} sources, sequential)"
        )
        return _run_sequential(other_tasks, label="commodities")

    failures = _run_parallel(
        [
            (_download_weather, "weather-group"),
            (_download_smard, "smard-group"),
            (_download_other, "commodities-group"),
        ],
        max_workers=3,
        label="all",
    )
    _exit_if_failures(failures)


# ── Update commands ─────────────────────────────────────────────────

update_app = typer.Typer(help="Incremental data update")
app.add_typer(update_app, name="update")


@update_app.command("all")
def update_all_sources():
    """Incremental update of all data sources.

    Same concurrency strategy as download: weather sequential,
    SMARD and commodities parallel, all three groups concurrent.
    """
    from energy_forecasting.config.smard import NATIONAL_REGIONS, TSO_REGIONS
    from energy_forecasting.data.commodities import all_commodity_sources
    from energy_forecasting.data.sources import EnergyChartsSource, SmardSource
    from energy_forecasting.data.weather import OpenMeteoSource

    smard_tasks = []
    for region in NATIONAL_REGIONS:
        s = SmardSource(region)
        smard_tasks.append((s.update, f"SMARD/{region}"))
    for tso_name in TSO_REGIONS:
        s = SmardSource(tso_name)
        smard_tasks.append((s.update, f"SMARD/{tso_name}"))

    weather_tasks = []
    for at in ["offshore", "onshore", "solar", "cities"]:
        for tso_name in TSO_REGIONS:
            source = OpenMeteoSource(at, tso_name)
            if source.locations:
                weather_tasks.append((source.update, f"weather/{at}/{tso_name}"))

    commodity_sources = all_commodity_sources()
    other_tasks = [(s.update, type(s).__name__) for s in commodity_sources]
    other_tasks.append((EnergyChartsSource().update, "EnergyCharts"))

    def _update_smard():
        logger.info(f"Updating SMARD ({len(smard_tasks)} regions in parallel)")
        return _run_parallel(smard_tasks, max_workers=len(smard_tasks), label="SMARD")

    def _update_weather():
        logger.info(f"Updating weather ({len(weather_tasks)} sources, sequential)")
        return _run_sequential(weather_tasks, label="weather")

    def _update_other():
        # Sequential: yfinance is not thread-safe (concurrent calls corrupt results)
        logger.info(
            f"Updating commodities + Energy Charts ({len(other_tasks)} sources, sequential)"
        )
        return _run_sequential(other_tasks, label="commodities")

    failures = _run_parallel(
        [
            (_update_weather, "weather-group"),
            (_update_smard, "smard-group"),
            (_update_other, "commodities-group"),
        ],
        max_workers=3,
        label="all",
    )
    _exit_if_failures(failures)


@update_app.command("smard")
def update_smard():
    """Update national + per-TSO SMARD data."""
    from energy_forecasting.config.smard import NATIONAL_REGIONS, TSO_REGIONS
    from energy_forecasting.data.sources import SmardSource

    tasks = []
    for region in NATIONAL_REGIONS:
        s = SmardSource(region)
        tasks.append((s.update, f"SMARD/{region}"))
    for tso_name in TSO_REGIONS:
        s = SmardSource(tso_name)
        tasks.append((s.update, f"SMARD/{tso_name}"))
    _exit_if_failures(_run_parallel(tasks, max_workers=len(tasks), label="SMARD"))


@update_app.command("weather")
def update_weather():
    """Update all weather data."""
    from energy_forecasting.config.smard import TSO_REGIONS
    from energy_forecasting.data.weather import OpenMeteoSource

    tasks = []
    for at in ["offshore", "onshore", "solar", "cities"]:
        for tso_name in TSO_REGIONS:
            source = OpenMeteoSource(at, tso_name)
            if source.locations:
                tasks.append((source.update, f"weather/{at}/{tso_name}"))
    _exit_if_failures(_run_sequential(tasks, label="weather"))


@update_app.command("commodities")
def update_commodities():
    """Update commodity sources."""
    from energy_forecasting.data.commodities import all_commodity_sources

    # Sequential: yfinance is not thread-safe (concurrent calls corrupt results)
    sources = all_commodity_sources()
    tasks = [(s.update, type(s).__name__) for s in sources]
    _exit_if_failures(_run_sequential(tasks, label="commodities"))


# ── Processing commands ────────────────────────────────────────────


@app.command("process")
def process(
    output: Path = typer.Option(None, help="Output path for merged.parquet"),
):
    """Clean and merge raw data into processed/merged.parquet + processed/tso/."""
    from energy_forecasting.data.merge import run_merge_pipeline

    run_merge_pipeline(output_path=output)


# ── Train commands ─────────────────────────────────────────────────

train_app = typer.Typer(help="Train models")
app.add_typer(train_app, name="train")


@train_app.command("gen-load")
def train_gen_load(
    target: str = typer.Option(
        None, help="Single target (wind_onshore, wind_offshore, solar, load)"
    ),
    region: str = typer.Option(None, help="Single region (DE_50HZ, DE_AMPRION, ...)"),
    model_type: str = typer.Option(
        None, help="Single model type (LGBMRegressor, XGBRegressor, ElasticNet)"
    ),
    trials: int = typer.Option(70, help="Optuna trials per model"),
    skip_ensemble: bool = typer.Option(False, "--skip-ensemble", help="Skip stacking ensemble"),
    reuse_params: bool = typer.Option(
        False,
        "--reuse-params",
        help=(
            "Skip Optuna search; reuse winning hyperparameters from each "
            "combo's most recent finished MLflow run. Used to extend OOF "
            "coverage (e.g. after bumping GEN_LOAD_HISTORICAL_FOLDS) "
            "without paying for re-search. Errors if any combo lacks a "
            "prior run."
        ),
    ),
    parallel: int = typer.Option(
        1,
        "--parallel",
        help=(
            "Parallel workers across (target, region, model) combos within "
            "each dependency wave. Each worker is allocated cpu_count() // "
            "parallel threads (env vars + LGBM/XGB n_jobs). Default 1 = "
            "sequential, identical to historical behavior."
        ),
    ),
):
    """Train gen/load models with Optuna HPO + stacking ensemble.

    Respects dependency order: wind/solar first, then load (uses wind/solar
    forecasts as features), then gen_load_diff (uses all).

    With ``--parallel N`` (N > 1), all (target, region, model) combos within
    a dependency wave are trained concurrently in subprocess workers, with
    thread counts shared evenly. Ensembles + historical-forecast export
    remain sequential per (target, region) after the wave's training jobs
    finish (they read the just-written MLflow artifacts).

    With ``--reuse-params``, Optuna search is skipped and each combo's
    winning hyperparameters are loaded from the most recent finished
    MLflow run. The final training pass still runs at full
    ``GEN_LOAD_HISTORICAL_FOLDS`` weekly folds.
    """
    import os
    import warnings

    warnings.filterwarnings("ignore")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from energy_forecasting.config.modeling import GEN_LOAD_TARGETS, GEN_LOAD_TRAINING_ORDER
    from energy_forecasting.modeling.gen_load import (
        ensemble_gen_load,
        retrain_gen_load_from_existing,
        train_gen_load_model,
    )
    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking

    ensure_mlflow_tracking()

    model_types = [model_type] if model_type else ["LGBMRegressor", "XGBRegressor", "ElasticNet"]

    # Group combos by dependency wave so each wave can run in parallel
    # without crossing target dependencies (load needs wind/solar trained
    # first; gen_load_diff needs everything).
    waves: list[list[tuple[str, str]]] = []
    for group in GEN_LOAD_TRAINING_ORDER:
        wave_combos: list[tuple[str, str]] = []
        for t in group:
            if target and t != target:
                continue
            if t not in GEN_LOAD_TARGETS:
                continue
            for r in GEN_LOAD_TARGETS[t]["regions"]:
                if region and r != region:
                    continue
                wave_combos.append((t, r))
        if wave_combos:
            waves.append(wave_combos)
    ordered_combos = [c for wave in waves for c in wave]

    total_cores = os.cpu_count() or 1
    threads_per_worker = max(1, total_cores // max(1, parallel))
    # Preserve historical "use all cores" behavior when parallel == 1.
    n_jobs_for_run = -1 if parallel <= 1 else threads_per_worker

    total = len(ordered_combos) * len(model_types)
    logger.info(
        f"Training: {len(ordered_combos)} target/region × {len(model_types)} models "
        f"= {total} runs, {trials} trials each. parallel={parallel}, "
        f"threads/worker={threads_per_worker}"
    )

    results: dict[str, str] = {}
    failures: list[str] = []

    for wave_idx, wave in enumerate(waves):
        wave_jobs: list[tuple[str, str, str, int, int, bool]] = [
            (t, r, mt, trials, n_jobs_for_run, reuse_params)
            for (t, r) in wave
            for mt in model_types
        ]
        mode = "reuse-params" if reuse_params else "Optuna search"
        logger.info(f"── Wave {wave_idx + 1}/{len(waves)}: {len(wave_jobs)} jobs ({mode}) ──")

        run_ids_by_combo: dict[tuple[str, str], dict[str, str]] = {(t, r): {} for (t, r) in wave}

        if parallel > 1:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # Use 'spawn' so each worker is a fresh Python process. With the
            # default 'fork' on Linux, workers inherit the parent's already-
            # initialised numpy/MKL/OpenMP thread pools, and the env vars set
            # by `_init_pool_worker` come too late to take effect — workers
            # end up using all 16 cores each, badly oversubscribing the box.
            # 'spawn' re-imports modules in the worker AFTER the initializer
            # runs, so OMP_NUM_THREADS et al. are honoured.
            spawn_ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=parallel,
                initializer=_init_pool_worker,
                initargs=(threads_per_worker,),
                mp_context=spawn_ctx,
            ) as executor:
                fut_to_label = {
                    executor.submit(_train_one_combo, job): f"{job[0]}/{job[1]}/{job[2]}"
                    for job in wave_jobs
                }
                for fut in as_completed(fut_to_label):
                    label = fut_to_label[fut]
                    t, r, mt, run_id, err = fut.result()
                    if run_id is not None:
                        results[label] = run_id
                        run_ids_by_combo[(t, r)][mt] = run_id
                        logger.info(f"✓ {label}: {run_id[:8]}")
                    else:
                        failures.append(label)
                        logger.error(f"FAILED: {label}\n{err}")
        else:
            # Sequential path — matches the original loop exactly.
            for n, (t, r, mt, _, n_jobs, reuse) in enumerate(wave_jobs, start=1):
                label = f"{t}/{r}/{mt}"
                logger.info(f"[wave {wave_idx + 1} {n}/{len(wave_jobs)}] {label}")
                try:
                    if reuse:
                        run_id = retrain_gen_load_from_existing(
                            target=t,
                            region=r,
                            model_type=mt,
                            n_jobs=n_jobs,
                        )
                    else:
                        run_id = train_gen_load_model(
                            target=t,
                            region=r,
                            model_type=mt,
                            optuna_trials=trials,
                            n_jobs=n_jobs,
                        )
                    results[label] = run_id
                    run_ids_by_combo[(t, r)][mt] = run_id
                except Exception:
                    logger.exception(f"FAILED: {label}")
                    failures.append(label)

        # Per-(target, region) ensemble + historical-forecasts export. These
        # are sequential because they read freshly-written MLflow artifacts
        # and run quickly (~30 s each); not worth process-pooling.
        for t, r in wave:
            base_run_metrics = run_ids_by_combo[(t, r)]
            base_run_ids = list(base_run_metrics.values())

            ensemble_run_id: str | None = None
            if not skip_ensemble and len(base_run_ids) >= 2:
                try:
                    ens_id = ensemble_gen_load(t, r, base_run_ids)
                    results[f"{t}/{r}/ensemble"] = ens_id
                    ensemble_run_id = ens_id
                except Exception:
                    logger.exception(f"FAILED: {t}/{r}/ensemble")
                    failures.append(f"{t}/{r}/ensemble")

            chosen_run_id = ensemble_run_id or _pick_best_base_run(
                t,
                r,
                base_run_metrics,
            )
            if chosen_run_id is not None:
                try:
                    _export_historical_forecasts(t, r, chosen_run_id)
                except Exception:
                    logger.exception(f"FAILED: historical_forecasts export for {t}/{r}")
                    failures.append(f"{t}/{r}/historical_forecasts")

                try:
                    best_base_id = _pick_best_base_run(t, r, base_run_metrics)
                    if best_base_id:
                        _write_gen_load_config(t, r, best_base_id)
                except Exception:
                    logger.exception(f"FAILED: gen_load_config.json write for {t}/{r}")
                    failures.append(f"{t}/{r}/gen_load_config")

    # National aggregates for per-TSO targets — sum predictions across regions.
    # Mirrors EMA's `export_national_forecasts.py` collapse step.
    try:
        _aggregate_national_historical_forecasts(ordered_combos)
    except Exception:
        logger.exception("FAILED: national aggregate of historical_forecasts")
        failures.append("national_historical_forecasts")

    logger.info(f"Done: {len(results)} succeeded, {len(failures)} failed")
    if failures:
        logger.warning(f"Failures: {failures}")
        raise typer.Exit(code=1)


def _init_pool_worker(threads_per_worker: int) -> None:
    """Limit thread counts in each ProcessPoolExecutor worker.

    Must run before the worker imports numpy/scikit-learn/lightgbm/xgboost,
    otherwise BLAS thread pools (MKL, OpenBLAS, OpenMP) are sized once at
    import time and never re-read these env vars. We rely on the executor
    being created with the 'spawn' start method (set in ``train_gen_load``)
    so this initializer runs in a fresh interpreter before any heavy
    imports — with the default 'fork' the parent's already-imported BLAS
    state would be inherited and the limits would be ignored.

    The model constructors also receive ``n_jobs=threads_per_worker``
    directly via the job tuple (LGBM/XGB read this rather than the env);
    this initializer primarily protects implicit BLAS calls in
    scikit-learn / pandas (StandardScaler, ElasticNet preprocessing, etc.)
    and any joblib parallelism in MAPIE.
    """
    import os

    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
    os.environ["BLIS_NUM_THREADS"] = str(threads_per_worker)
    # Cap joblib (used by sklearn / MAPIE for inner CV parallelism) to the
    # same slice so it can't multiply contention.
    os.environ["LOKY_MAX_CPU_COUNT"] = str(threads_per_worker)


def _train_one_combo(
    job: tuple[str, str, str, int, int, bool],
) -> tuple[str, str, str, str | None, str | None]:
    """ProcessPoolExecutor worker. Trains one (target, region, model) combo.

    The job tuple is ``(target, region, model_type, trials, n_jobs,
    reuse_params)``. Returns ``(target, region, model_type, run_id,
    error_traceback)``. On success ``error_traceback`` is None; on failure
    ``run_id`` is None and ``error_traceback`` carries the formatted
    traceback for parent-process logging.
    """
    target, region, model_type, trials, n_jobs, reuse_params = job
    import traceback
    import warnings

    warnings.filterwarnings("ignore")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from energy_forecasting.modeling.gen_load import (
        retrain_gen_load_from_existing,
        train_gen_load_model,
    )

    try:
        if reuse_params:
            run_id = retrain_gen_load_from_existing(
                target=target,
                region=region,
                model_type=model_type,
                n_jobs=n_jobs,
            )
        else:
            run_id = train_gen_load_model(
                target=target,
                region=region,
                model_type=model_type,
                optuna_trials=trials,
                n_jobs=n_jobs,
            )
        return (target, region, model_type, run_id, None)
    except Exception:
        return (target, region, model_type, None, traceback.format_exc())


def _pick_best_base_run(
    target: str,
    region: str,
    base_runs: dict[str, str],
) -> str | None:
    """Pick the base run with the lowest CV MAE.

    Uses ``cv_mae`` (40 folds × 168h = ~9.5 months of evaluations) rather
    than holdout MAE (168h, noisy). Matches EMA's 2026-03-15 fix that
    switched from ``best_model_forecast.json`` (training-data evaluation)
    to ``best_model.json`` (rolling CV).

    Falls back to holdout MAE, then to the first available run.
    """
    if not base_runs:
        return None
    import mlflow

    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking

    ensure_mlflow_tracking()
    client = mlflow.MlflowClient()
    by_cv: list[tuple[float, str]] = []
    by_holdout: list[tuple[float, str]] = []
    for _mt, run_id in base_runs.items():
        try:
            metrics = client.get_run(run_id).data.metrics
            cv = metrics.get("cv_mae")
            holdout = metrics.get("mae")
            if cv is not None:
                by_cv.append((cv, run_id))
            if holdout is not None:
                by_holdout.append((holdout, run_id))
        except Exception:
            continue
    if by_cv:
        return min(by_cv)[1]
    if by_holdout:
        return min(by_holdout)[1]
    return next(iter(base_runs.values()))


def _export_historical_forecasts(
    target: str,
    region: str,
    run_id: str,
) -> None:
    """Concatenate OOF + holdout predictions for one run and save to disk.

    Output: ``data/processed/historical_forecasts/{target}_{region}.parquet``.
    Schema: y_true, y_pred, y_lower, y_upper (NaN for runs that bypassed
    MAPIE — i.e. recursive-lag models).

    Replaces EMA's ``generate_historical_forecasts.py`` for the merged repo:
    OOF folds + holdout already produce the same multi-week leak-free
    forecast time series as a byproduct of training.
    """
    import mlflow
    import pandas as pd

    from energy_forecasting.config import PROCESSED_DATA_DIR
    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking

    ensure_mlflow_tracking()

    out_dir = PROCESSED_DATA_DIR / "historical_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = mlflow.MlflowClient()
    parts: list[pd.DataFrame] = []
    for name in ("oof_predictions.parquet", "holdout_predictions.parquet"):
        try:
            local = client.download_artifacts(run_id, f"predictions/{name}")
        except Exception:
            logger.warning(f"{target}/{region}: missing artifact {name} on run {run_id}")
            continue
        df = pd.read_parquet(local)
        parts.append(df)

    if not parts:
        logger.warning(
            f"{target}/{region}: no prediction artifacts found on run {run_id}; "
            f"skipping historical_forecasts export"
        )
        return

    combined = pd.concat(parts, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    for col in ("y_lower", "y_upper"):
        if col not in combined.columns:
            combined[col] = float("nan")
    combined = combined[["y_true", "y_pred", "y_lower", "y_upper"]]

    out_path = out_dir / f"{target}_{region}.parquet"
    combined.to_parquet(out_path)
    logger.info(
        f"Saved historical_forecasts/{target}_{region}.parquet: "
        f"{combined.shape[0]} rows, span "
        f"{combined.index[0]} → {combined.index[-1]}"
    )


def _aggregate_national_historical_forecasts(
    ordered_combos: list[tuple[str, str]],
) -> None:
    """Sum per-TSO historical_forecasts into national aggregates.

    For each per-TSO target (wind_onshore, wind_offshore, solar, load),
    sums y_pred (and y_true where available) across the regions present
    in this run. Saves to
    ``data/processed/historical_forecasts/{target}_DE_NATIONAL.parquet``
    if it does not already exist (gen_load_diff is national directly and
    is left untouched).
    """
    import pandas as pd

    from energy_forecasting.config import PROCESSED_DATA_DIR

    out_dir = PROCESSED_DATA_DIR / "historical_forecasts"
    if not out_dir.exists():
        return

    by_target: dict[str, list[str]] = {}
    for t, r in ordered_combos:
        if t == "gen_load_diff":
            continue
        by_target.setdefault(t, []).append(r)

    for t, regions in by_target.items():
        if len(regions) < 2:
            continue  # nothing to aggregate
        frames: list[pd.DataFrame] = []
        for r in regions:
            path = out_dir / f"{t}_{r}.parquet"
            if not path.exists():
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            continue
        common_idx = frames[0].index
        for df in frames[1:]:
            common_idx = common_idx.intersection(df.index)
        if len(common_idx) == 0:
            logger.warning(
                f"{t}: no overlapping index across regions, skipping national aggregate"
            )
            continue
        agg = sum(df.reindex(common_idx) for df in frames)
        agg = agg[["y_true", "y_pred", "y_lower", "y_upper"]]
        nat_path = out_dir / f"{t}_DE_NATIONAL.parquet"
        agg.to_parquet(nat_path)
        logger.info(
            f"Saved historical_forecasts/{t}_DE_NATIONAL.parquet (sum of "
            f"{len(frames)} regions): {agg.shape[0]} rows"
        )


@app.command("features")
def features(
    feature_list: str = typer.Option("slim", help="Feature list: slim, full, gen_load"),
    input_path: Path = typer.Option(None, help="Input merged.parquet path"),
    output: Path = typer.Option(None, help="Output features parquet path"),
    validate_only: bool = typer.Option(
        False, "--validate-only", help="Only validate, don't compute"
    ),
):
    """Compute feature matrix from merged data using the suffix DSL."""
    from energy_forecasting.config import PROCESSED_DATA_DIR
    from energy_forecasting.config.features import (
        GEN_LOAD_FEATURES,
        PRICE_FEATURES_FULL,
        PRICE_FEATURES_MAX,
        PRICE_FEATURES_SLIM,
    )
    from energy_forecasting.data.io import load_parquet, save_parquet
    from energy_forecasting.features.engine import engineer_features
    from energy_forecasting.features.validation import validate_features

    lists = {
        "slim": PRICE_FEATURES_SLIM,
        "full": PRICE_FEATURES_FULL,
        "max": PRICE_FEATURES_MAX,
        "gen_load": GEN_LOAD_FEATURES,
    }
    if feature_list not in lists:
        raise typer.BadParameter(f"Unknown feature list: {feature_list}. Use: {list(lists)}")

    features = lists[feature_list]
    logger.info(f"Feature list '{feature_list}': {len(features)} features")

    # Validate
    errors = validate_features(features)
    if errors:
        for e in errors:
            logger.error(f"  {e.feature_str}: {e.reason}")
        raise typer.Exit(code=1)
    logger.info("Validation passed")

    if validate_only:
        return

    # Load merged data
    merged_path = input_path or PROCESSED_DATA_DIR / "merged.parquet"
    df = load_parquet(merged_path)
    logger.info(f"Loaded {len(df)} rows from {merged_path}")

    # Compute
    result = engineer_features(df, features, validate=False)

    # Save
    out_path = output or PROCESSED_DATA_DIR / f"features_{feature_list}.parquet"
    save_parquet(result, out_path)
    logger.info(f"Saved {result.shape} to {out_path}")


@train_app.command("price")
def train_price(
    feature_versions: str = typer.Option(
        "max",
        "--feature-versions",
        help="Comma-separated feature_version list (slim, full, max). "
        "Default 'max'; use 'slim' for a quick wiring check. "
        "Ignored when --feature-selection is set.",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Quick smoke run: feature_versions=slim, three tree families "
        "+ Ridge (omits Lasso to keep wall time short).",
    ),
    feature_selection: bool = typer.Option(
        False,
        "--feature-selection",
        help="Run feature_selection on the MAX dataset, then tune over the "
        "top-K candidate feature sets discovered.",
    ),
    top_k: int = typer.Option(
        4,
        "--top-k",
        help="Number of feature_selection candidates to feed tuning.",
    ),
    use_rfecv: bool = typer.Option(
        False,
        "--use-rfecv",
        help="Include RFECV in feature_selection. Slowest step (~1-2h with default settings).",
    ),
    pin_feature_version: str = typer.Option(
        None,
        "--pin-feature-version",
        help="Skip feature selection and tune only on the named dataset "
        "(e.g. 'fs_shap_top60'). Use for ongoing retrains once a feature "
        "set has been chosen from a research run.",
    ),
):
    """End-to-end price pipeline: tune → retrain → ensemble bake-off → write ensemble_config.json."""
    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking
    from energy_forecasting.modeling.price import run_price_pipeline
    from energy_forecasting.modeling.tuning import PRICE_LINEAR_TYPES, PRICE_TREE_TYPES

    ensure_mlflow_tracking()

    if quick:
        # Quick smoke: slim features, all three tree families, plus Ridge.
        fv_list = ["slim"]
        tree_types = ("LGBMRegressor", "XGBRegressor", "CatBoostRegressor")
        linear_types = ("Ridge",)
    else:
        fv_list = [v.strip() for v in feature_versions.split(",") if v.strip()]
        tree_types = PRICE_TREE_TYPES
        linear_types = PRICE_LINEAR_TYPES

    if pin_feature_version is not None:
        # Ongoing-retrain mode — skip feature selection, tune only on the
        # pinned dataset. The dataset must already exist (built during a
        # previous research run).
        from energy_forecasting.modeling.datasets import find_dataset

        if not find_dataset(f"price_{pin_feature_version}"):
            raise typer.BadParameter(
                f"Dataset 'price_{pin_feature_version}' not found. "
                "Run a research pass first (--feature-selection)."
            )
        fv_list = [pin_feature_version]
        feature_selection = False
        logger.info(
            f"Price pipeline (pinned): feature_version={pin_feature_version}, "
            f"trees={tree_types}, linear={linear_types}"
        )
    elif feature_selection:
        logger.info(
            f"Price pipeline: feature_selection on MAX, top_k={top_k}, "
            f"rfecv={use_rfecv}, trees={tree_types}, linear={linear_types}"
        )
    else:
        logger.info(
            f"Price pipeline: feature_versions={fv_list}, "
            f"trees={tree_types}, linear={linear_types}"
        )
    result = run_price_pipeline(
        feature_versions=fv_list,
        tree_types=tree_types,
        linear_types=linear_types,
        use_feature_selection=feature_selection,
        feature_selection_top_k=top_k,
        feature_selection_use_rfecv=use_rfecv,
    )
    logger.info(
        f"Done. Ensemble winner: {result['ensemble_method']} "
        f"(holdout MAE={result['metrics']['mae']:.3f})"
    )
    logger.info(f"Config: {result['config_path']}")


# ── gen_load_config.json helpers ────────────────────────────────────


def _write_gen_load_config(target: str, region: str, run_id: str) -> None:
    """Write/update this combo's entry in models/gen_load_config.json.

    Loads best_config.json from the run's MLflow artifact and stores the
    weather_config + dataset_params + run_id needed for inference.
    """
    import json
    from datetime import datetime, timezone

    import mlflow

    from energy_forecasting.config import MODELS_DIR
    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking

    ensure_mlflow_tracking()
    client = mlflow.MlflowClient()
    config_path = client.download_artifacts(run_id, "optuna/best_config.json")
    with open(config_path) as f:
        best_config = json.load(f)

    config_file = MODELS_DIR / "gen_load_config.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        with open(config_file) as f:
            gl_config = json.load(f)
    else:
        gl_config = {"generated_at": "", "combos": {}}

    combo_key = f"{target}/{region}"
    gl_config["combos"][combo_key] = {
        "run_id": run_id,
        "model_type": best_config["model_type"],
        "model_params": best_config["model_params"],
        "weather_config": best_config["weather_config"],
        "dataset_params": best_config["dataset_params"],
    }
    gl_config["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(config_file, "w") as f:
        json.dump(gl_config, f, indent=2, default=str)
    logger.info(f"Updated gen_load_config.json: {combo_key} → {run_id[:8]}")


# ── Deploy commands ─────────────────────────────────────────────────

deploy_app = typer.Typer(help="Inference, export and serving")
app.add_typer(deploy_app, name="deploy")


@deploy_app.command("gen-load-config")
def gen_load_config_cmd(
    force: bool = typer.Option(False, "--force", help="Overwrite existing entries"),
):
    """Scan MLflow for best gen/load runs and write models/gen_load_config.json.

    Run this once after training to bootstrap the config file if the training
    run that wrote it is no longer available.
    """

    from energy_forecasting.config.modeling import GEN_LOAD_TARGETS
    from energy_forecasting.modeling.mlflow_utils import ensure_mlflow_tracking

    ensure_mlflow_tracking()

    from energy_forecasting.deploy.model_store import (
        GEN_LOAD_CONFIG_PATH,
        load_gen_load_config,
    )

    if GEN_LOAD_CONFIG_PATH.exists() and not force:
        existing = load_gen_load_config()
        logger.info(
            f"gen_load_config.json already has {len(existing['combos'])} combos. "
            "Use --force to overwrite."
        )
        return

    for target, info in GEN_LOAD_TARGETS.items():
        for region in info["regions"]:
            from energy_forecasting.modeling.gen_load import _find_latest_base_run

            model_types = ["LGBMRegressor", "XGBRegressor"]
            run_ids = {}
            for mt in model_types:
                try:
                    run_ids[mt] = _find_latest_base_run(target, region, mt)
                except Exception:
                    pass
            if not run_ids:
                logger.warning(f"No runs found for {target}/{region}, skipping")
                continue
            best_id = _pick_best_base_run(target, region, run_ids)
            if best_id:
                try:
                    _write_gen_load_config(target, region, best_id)
                except Exception:
                    logger.exception(f"Failed for {target}/{region}")

    logger.info("gen_load_config.json written.")


@deploy_app.command("export-models")
def export_models_cmd(
    gen_load: bool = typer.Option(True, "--gen-load/--no-gen-load", help="Export gen/load models"),
    price: bool = typer.Option(True, "--price/--no-price", help="Export price models"),
):
    """Export production models from MLflow to disk (models/gen_load/, models/price/)."""
    from energy_forecasting.deploy.model_store import (
        export_gen_load_models,
        export_price_models,
    )

    if gen_load:
        written = export_gen_load_models()
        logger.info(f"Gen/load: {len(written)} models exported")
    if price:
        written = export_price_models()
        logger.info(f"Price: {len(written)} models exported")


@deploy_app.command("forecast")
def forecast_cmd(
    skip_update: bool = typer.Option(
        False, "--skip-update", help="Skip data update (use existing data)"
    ),
):
    """Run the full daily inference pipeline: update → gen/load → price → validate → write."""
    from energy_forecasting.deploy.inference import run_inference

    result = run_inference(skip_update=skip_update)
    price_df = result["price"]
    logger.info(
        f"Forecast complete. Price: {len(price_df)} hours, "
        f"mean={price_df['y_pred'].mean():.1f} EUR/MWh"
    )


@deploy_app.command("narrative")
def narrative_cmd(model: str = typer.Option("llama-3.3-70b-versatile")):
    """Generate the AI narrative for tomorrow's forecast (non-blocking; degrades gracefully)."""
    from energy_forecasting.deploy.narrative import generate_forecast_narrative

    result = generate_forecast_narrative(model=model)
    logger.info(
        f"Narrative status: {result['status']}"
        + (f" ({result['reason']})" if result.get("reason") else "")
    )


@deploy_app.command("narrative-yearly")
def narrative_yearly_cmd(model: str = typer.Option("llama-3.3-70b-versatile")):
    """Generate the weekly AI yearly-recap narrative (non-blocking; degrades gracefully)."""
    from energy_forecasting.deploy.narrative import generate_yearly_narrative

    result = generate_yearly_narrative(model=model)
    logger.info(
        f"Yearly narrative status: {result['status']}"
        + (f" ({result['reason']})" if result.get("reason") else "")
    )


@deploy_app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the FastAPI forecast server."""
    import uvicorn

    uvicorn.run(
        "energy_forecasting.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@deploy_app.command("retrain")
def retrain_cmd(
    price_only: bool = typer.Option(
        False, "--price-only", help="Retrain only price models (not gen/load)"
    ),
    force: bool = typer.Option(False, "--force", help="Retrain even if no degradation detected"),
    holdout_days: int = typer.Option(
        None, "--holdout-days", help="Override holdout days for degradation check"
    ),
):
    """Retrain price ensemble. Gen/load retrain must be run manually (8-12 hours)."""
    from energy_forecasting.deploy.retrain import run_retrain

    result = run_retrain(price_only=price_only, force=force, holdout_days=holdout_days)
    if result.get("needs_reselection"):
        logger.warning(
            "needs_reselection=True: new ensemble MAE degraded beyond threshold. "
            "Run 'energy-forecasting train price --feature-selection' to reselect candidates."
        )
    else:
        logger.info(
            f"Retrain complete. New MAE: {result.get('new_mae', '?'):.3f}, "
            f"previous: {result.get('old_mae', '?'):.3f}"
        )
