"""Daily inference pipeline orchestrator.

Wires together data update → gen/load inference → price inference →
validation → output writing.

Usage (CLI):
    energy-forecasting deploy forecast
    energy-forecasting deploy forecast --skip-update

Usage (Python):
    from energy_forecasting.deploy.inference import run_inference
    result = run_inference()
"""

from __future__ import annotations

from loguru import logger


def _update_data() -> None:
    """Run incremental data update (SMARD, weather, commodities)."""
    from energy_forecasting.config.smard import NATIONAL_REGIONS, TSO_REGIONS
    from energy_forecasting.data.commodities import all_commodity_sources
    from energy_forecasting.data.sources import EnergyChartsSource, SmardSource
    from energy_forecasting.data.weather import OpenMeteoSource

    smard_tasks = [(SmardSource(r).update, f"SMARD/{r}") for r in NATIONAL_REGIONS] + [
        (SmardSource(t).update, f"SMARD/{t}") for t in TSO_REGIONS
    ]
    weather_tasks = [
        (OpenMeteoSource(at, t).update, f"weather/{at}/{t}")
        for at in ["offshore", "onshore", "solar", "cities"]
        for t in TSO_REGIONS
        if OpenMeteoSource(at, t).locations
    ]
    other_tasks = [(s.update, type(s).__name__) for s in all_commodity_sources()] + [
        (EnergyChartsSource().update, "EnergyCharts")
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from energy_forecasting.data.weather import RateLimitExhausted

    def _par(tasks, max_workers):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(fn): name for fn, name in tasks}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception:
                    logger.exception(f"Failed: {futs[f]}")

    def _seq(tasks):
        for fn, name in tasks:
            try:
                fn()
            except RateLimitExhausted as exc:
                logger.warning(f"Rate limit on {name}: {exc}")
                return
            except Exception:
                logger.exception(f"Failed: {name}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [
            pool.submit(_par, smard_tasks, len(smard_tasks)),
            pool.submit(_seq, weather_tasks),
            pool.submit(_seq, other_tasks),
        ]
        for f in futs:
            f.result()


def _merge_and_process() -> None:
    """Rerun the merge pipeline on updated raw data."""
    from energy_forecasting.config import PROCESSED_DATA_DIR
    from energy_forecasting.data.merge import run_merge_pipeline

    logger.info("Rebuilding merged dataset...")
    run_merge_pipeline(output_path=PROCESSED_DATA_DIR / "merged.parquet")
    logger.info("Merged dataset rebuilt")


def run_inference(skip_update: bool = False) -> dict:
    """Run the full daily inference pipeline.

    Parameters
    ----------
    skip_update : bool
        If True, skip data update and merge steps (use when data was already
        updated in a previous step, e.g. in GitHub Actions collect-data job).

    Returns
    -------
    dict with keys:
        price     : 24-row DataFrame [y_pred, y_lower, y_upper]
        gen_load  : dict[(target, region) → 168-row DataFrame]
    """
    errors: dict[str, Exception] = {}

    # 1. Data update
    if not skip_update:
        try:
            logger.info("Updating data sources...")
            _update_data()
        except Exception as exc:
            logger.exception("Data update failed — proceeding with existing data")
            errors["data_update"] = exc
        try:
            _merge_and_process()
        except Exception as exc:
            logger.exception("Merge step failed — proceeding with existing merged data")
            errors["merge"] = exc

    # 2. Gen/load inference
    from energy_forecasting.deploy.gen_load_inference import (
        run_gen_load_inference,
        update_historical_forecasts,
    )

    logger.info("Running gen/load inference...")
    gen_load_results = run_gen_load_inference()
    update_historical_forecasts(gen_load_results)

    # 3. Price inference (uses the just-updated historical_forecasts)
    from energy_forecasting.deploy.price_inference import run_price_inference

    logger.info("Running price inference...")
    price_df = run_price_inference()

    # 4. Validate — fail hard before any output is written
    from energy_forecasting.deploy.validation import validate_outputs

    validate_outputs(price_df, gen_load_results)

    # 5. Write outputs
    from energy_forecasting.deploy.publish import compute_errors, write_outputs

    try:
        write_outputs(price_df, gen_load_results)
        compute_errors(price_df)
    except Exception as exc:
        logger.exception("Output writing failed")
        errors["publish"] = exc

    if errors:
        logger.warning(
            f"Pipeline complete with {len(errors)} non-fatal error(s): {list(errors.keys())}"
        )
    else:
        logger.info("Daily inference pipeline complete")

    return {"price": price_df, "gen_load": gen_load_results}
