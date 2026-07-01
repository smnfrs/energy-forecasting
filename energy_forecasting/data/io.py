"""Parquet I/O utilities with optional compression and dtype reduction.

Ported from EMA's ParquetOperations class, simplified to two functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def reduce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 -> float32, int64 -> int32 where values fit.

    Returns a new DataFrame with reduced dtypes. Does not mutate the input.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        if df[col].isna().all():
            continue
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
            df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype("int32")
    return df


def save_parquet(
    df: pd.DataFrame,
    path: Path | str,
    compress: bool = True,
    downcast: bool = True,
) -> None:
    """Save DataFrame to Parquet with optional zstd compression and dtype reduction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if downcast:
        df = reduce_dtypes(df)

    compression = "zstd" if compress else None
    df.to_parquet(path, engine="pyarrow", compression=compression)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.debug(f"Saved {path.name}: {len(df)} rows, {size_mb:.1f} MB")


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load Parquet file."""
    return pd.read_parquet(Path(path), engine="pyarrow")
