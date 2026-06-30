"""Dataset management for model training.

Separates feature computation from training. Datasets are Parquet files on disk,
tracked in MLflow via mlflow.log_input() — not as runs or experiments.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

from energy_forecasting.features.engine import engineer_features, extend_features

DATASET_DIR = Path("data/processed/datasets")
TARGET_COL_SUFFIX = "__target"


def prepare_dataset(
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
    name: str,
) -> Path:
    """Compute features and save as a Parquet dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame (from data/processed/merged.parquet).
    feature_list : list[str]
        Feature strings in suffix DSL format.
    target_col : str
        Column name in df to use as the target variable.
    name : str
        Dataset name, e.g. "price_slim_v1". Used as the Parquet filename.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    features = engineer_features(df, feature_list)
    # Attach target as a column with a distinct suffix so load_dataset can split
    features[target_col + TARGET_COL_SUFFIX] = df[target_col]

    # Drop rows where target is NaN (can't train on them)
    target_key = target_col + TARGET_COL_SUFFIX
    features = features.dropna(subset=[target_key])

    path = DATASET_DIR / f"{name}.parquet"
    features.to_parquet(path)
    logger.info(f"Saved dataset '{name}': {features.shape} to {path}")
    return path


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load X (features) and y (target) from a Parquet dataset.

    The target column is identified by the TARGET_COL_SUFFIX marker.
    """
    df = pd.read_parquet(path)

    target_cols = [c for c in df.columns if c.endswith(TARGET_COL_SUFFIX)]
    if len(target_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 target column (ending with '{TARGET_COL_SUFFIX}'), "
            f"found {len(target_cols)}: {target_cols}"
        )

    target_col = target_cols[0]
    y = df[target_col]
    y.name = target_col.removesuffix(TARGET_COL_SUFFIX)
    X = df.drop(columns=[target_col])

    return X, y


def update_dataset(
    path: Path,
    df: pd.DataFrame,
    feature_list: list[str],
    target_col: str,
) -> Path:
    """Extend existing dataset with new rows.

    Loads the existing Parquet, computes features for new dates via
    extend_features(), and overwrites the file.
    """
    existing = pd.read_parquet(path)

    # Separate target from features for extension
    target_key = target_col + TARGET_COL_SUFFIX
    existing_features = existing.drop(columns=[target_key], errors="ignore")

    extended = extend_features(existing_features, df, feature_list)
    extended[target_key] = df[target_col].reindex(extended.index)
    extended = extended.dropna(subset=[target_key])

    extended.to_parquet(path)
    logger.info(f"Updated dataset at {path}: {extended.shape}")
    return path


def find_dataset(name: str) -> Path | None:
    """Check if a dataset Parquet file exists. Returns path or None."""
    path = DATASET_DIR / f"{name}.parquet"
    return path if path.exists() else None


def log_dataset_to_run(X: pd.DataFrame, path: Path):
    """Register a dataset with MLflow inside an active run.

    Called inside train_model() to record provenance.
    Name derived from path.stem.
    """
    dataset = mlflow.data.from_pandas(X, source=str(path), name=path.stem)
    mlflow.log_input(dataset, context="training")
