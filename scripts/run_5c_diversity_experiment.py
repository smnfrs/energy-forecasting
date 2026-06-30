"""Stage 5c diversity/accuracy experiment (2026-06-07).

Controlled, model-focused re-run that REUSES the four feature-selected
datasets from the 2026-05-29 run (no RFECV/SHAP recompute) so the only thing
that changes vs the 11.24 baseline is the model set + configs:

  * LGBM fixed     — num_leaves scaled to max_depth, bagging_freq=1, objective
                     kept at MAE for EP comparability (capacity bug fixed).
  * LGBMQuantile   — quantile objective (alpha=0.5) diversity candidate.
  * Huber          — robust-linear diversity candidate.
  * `max` dropped; target transforms dropped as a search axis.

The definitive question this answers: do LGBMQuantile and Huber earn any
weight in the SLSQP ensemble bake-off, or are they (as the pre-run probes
suggested) redundant / too weak to help?

Writes to models/ensemble_config_5c_diversity.json so the production
ensemble_config.json (MAE 11.24) is left untouched for comparison.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from energy_forecasting.modeling.price import run_price_pipeline

DATASET_DIR = Path("data/processed/datasets")
PRECOMPUTED = {
    "fs_rfecv_optimum": DATASET_DIR / "price_fs_rfecv_optimum.parquet",
    "fs_shap_top66": DATASET_DIR / "price_fs_shap_top66.parquet",
    "fs_shap_top90": DATASET_DIR / "price_fs_shap_top90.parquet",
    "fs_shap_top247": DATASET_DIR / "price_fs_shap_top247.parquet",
}
OUTPUT_CONFIG = Path("models/ensemble_config_5c_diversity.json")


def main() -> None:
    for name, path in PRECOMPUTED.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed dataset {name}: {path}")

    logger.info(
        "Launching 5c diversity experiment over "
        f"{len(PRECOMPUTED)} datasets: {list(PRECOMPUTED)}"
    )
    result = run_price_pipeline(
        feature_versions=[],
        precomputed_datasets=PRECOMPUTED,
        output_config=OUTPUT_CONFIG,
    )
    metrics = result["metrics"]
    logger.info(
        f"DONE. Ensemble method={result['ensemble_method']} "
        f"holdout MAE={metrics['mae']:.4f} "
        f"(baseline 11.24) -> {OUTPUT_CONFIG}"
    )


if __name__ == "__main__":
    main()
